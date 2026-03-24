const connectBtn = document.getElementById("connectBtn");
const disconnectBtn = document.getElementById("disconnectBtn");
const clearBtn = document.getElementById("clearBtn");
const socketState = document.getElementById("socketState");
const phaseState = document.getElementById("phaseState");
const micState = document.getElementById("micState");
const playbackState = document.getElementById("playbackState");
const userLive = document.getElementById("userLive");
const assistantLive = document.getElementById("assistantLive");
const transcriptList = document.getElementById("transcriptList");
const debugState = document.getElementById("debugState");
const metricsState = document.getElementById("metricsState");

const state = {
  assetVersion: "20260324e",
  controlWs: null,
  uplinkWs: null,
  audioWs: null,
  streamId: null,
  sessionId: null,
  audioContext: null,
  mediaStream: null,
  sourceNode: null,
  workletNode: null,
  browserSampleRate: 16000,
  playbackClock: 0,
  playbackNodes: new Set(),
  playerFrames: [],
  playerTask: null,
  activeResponseId: null,
  started: false,
  uplinkBound: false,
  audioBound: false,
  renderTimer: null,
  pendingDebugPayload: null,
  pendingMetricsPayload: null,
  liveText: { user: "...", assistant: "..." },
  llmFinalText: "",
  playbackAckStartedAt: 0,
};

const UPLINK_FRAME_MS = 40;
const MAX_UPLINK_BUFFER = 65536;
const MAX_PLAYBACK_BUFFER_MS = 200;
const PLAYER_FRAME_MS = 20;
const PLAYER_IDLE_WAIT_MS = 10;
const PLAYER_LEAD_MS = 40;

function wsUrl(path) {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}${path}`;
}

function setStatus(target, text) {
  target.textContent = text;
}

function escapeHtml(text) {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function scheduleRender() {
  if (state.renderTimer !== null) {
    return;
  }
  state.renderTimer = window.setTimeout(() => {
    state.renderTimer = null;
    userLive.textContent = state.liveText.user || "...";
    assistantLive.textContent = state.liveText.assistant || "...";
    if (state.pendingDebugPayload) {
      debugState.textContent = JSON.stringify(state.pendingDebugPayload, null, 2);
      state.pendingDebugPayload = null;
    }
    if (state.pendingMetricsPayload) {
      metricsState.textContent = JSON.stringify(state.pendingMetricsPayload, null, 2);
      state.pendingMetricsPayload = null;
    }
  }, 80);
}

function appendBubble(speaker, text) {
  if (!text.trim()) {
    return;
  }
  const bubble = document.createElement("article");
  bubble.className = `bubble ${speaker}`;
  bubble.innerHTML = `<span class="speaker label">${speaker}</span><p>${escapeHtml(text)}</p>`;
  transcriptList.appendChild(bubble);
  transcriptList.scrollTop = transcriptList.scrollHeight;
}

function clearTranscript() {
  transcriptList.innerHTML = "";
  debugState.textContent = "waiting for /turn events...";
  metricsState.textContent = "waiting for response metrics...";
  state.liveText = { user: "...", assistant: "..." };
  state.llmFinalText = "";
  state.pendingDebugPayload = null;
  state.pendingMetricsPayload = null;
  scheduleRender();
}

function packAudioFrame(header, pcmBuffer) {
  // Keep audio frames binary on the wire: a tiny JSON header plus raw PCM16.
  const headerBytes = new TextEncoder().encode(JSON.stringify(header));
  const packed = new Uint8Array(4 + headerBytes.length + pcmBuffer.byteLength);
  const view = new DataView(packed.buffer);
  view.setUint32(0, headerBytes.length, false);
  packed.set(headerBytes, 4);
  packed.set(new Uint8Array(pcmBuffer), 4 + headerBytes.length);
  return packed.buffer;
}

function unpackAudioFrame(arrayBuffer) {
  const bytes = new Uint8Array(arrayBuffer);
  const view = new DataView(arrayBuffer);
  const headerLen = view.getUint32(0, false);
  const headerBytes = bytes.slice(4, 4 + headerLen);
  const payload = bytes.slice(4 + headerLen).buffer;
  const header = JSON.parse(new TextDecoder().decode(headerBytes));
  return { header, payload };
}

function base64ToInt16(base64) {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return new Int16Array(bytes.buffer);
}

function splitPcmFrames(pcmBuffer, sampleRate, frameMs) {
  const bytesPerFrame = Math.max(2, Math.round((sampleRate * frameMs) / 1000) * 2);
  const frames = [];
  const bytes = new Uint8Array(pcmBuffer);
  for (let offset = 0; offset < bytes.length; offset += bytesPerFrame) {
    frames.push(bytes.slice(offset, Math.min(offset + bytesPerFrame, bytes.length)).buffer);
  }
  return frames;
}

function queuePcmChunk(pcmBuffer, sampleRate, responseId, generatedAtMs) {
  if (!state.audioContext) {
    return;
  }
  if (state.activeResponseId && responseId && responseId !== state.activeResponseId) {
    return;
  }
  state.activeResponseId = responseId;
  const queuedMs = state.playerFrames.length * PLAYER_FRAME_MS;
  if (queuedMs > MAX_PLAYBACK_BUFFER_MS) {
    console.debug("drop tts chunk due to deep buffer", { queuedMs, responseId });
    return;
  }
  const frames = splitPcmFrames(pcmBuffer, sampleRate, PLAYER_FRAME_MS);
  for (const frame of frames) {
    state.playerFrames.push({
      pcmBuffer: frame,
      sampleRate,
      responseId,
      generatedAtMs,
    });
  }
  if (generatedAtMs) {
    state.pendingMetricsPayload = {
      ...(state.pendingMetricsPayload || {}),
      downlink_audio_buffer_ms: Math.round(state.playerFrames.length * PLAYER_FRAME_MS),
      audio_chunk_age_ms: Math.round(Date.now() - generatedAtMs),
      queued_audio_chunks: state.playerFrames.length,
    };
    scheduleRender();
  }
  ensurePlaybackLoop();
}

function scheduleFrame(frame) {
  const samples = new Int16Array(frame.pcmBuffer);
  const buffer = state.audioContext.createBuffer(1, samples.length, frame.sampleRate);
  const channel = buffer.getChannelData(0);
  for (let i = 0; i < samples.length; i += 1) {
    channel[i] = samples[i] / 0x7fff;
  }
  const source = state.audioContext.createBufferSource();
  source.buffer = buffer;
  source.connect(state.audioContext.destination);
  const now = state.audioContext.currentTime;
  const leadSeconds = PLAYER_LEAD_MS / 1000;
  state.playbackClock = Math.max(state.playbackClock, now + leadSeconds);
  source.start(state.playbackClock);
  state.playbackClock += buffer.duration;
  source.onended = () => state.playbackNodes.delete(source);
  state.playbackNodes.add(source);
  setStatus(playbackState, "streaming");
}

function ensurePlaybackLoop() {
  if (state.playerTask) {
    return;
  }
  state.playerTask = window.setInterval(() => {
    if (!state.audioContext) {
      return;
    }
    if (!state.playerFrames.length) {
      if (state.playbackNodes.size === 0) {
        setStatus(playbackState, "idle");
      }
      return;
    }
    const queuedMs = Math.max(0, (state.playbackClock - state.audioContext.currentTime) * 1000);
    if (queuedMs > PLAYER_LEAD_MS + PLAYER_FRAME_MS) {
      return;
    }
    const frame = state.playerFrames.shift();
    if (!frame) {
      return;
    }
    scheduleFrame(frame);
  }, PLAYER_IDLE_WAIT_MS);
}

function clearPlayback() {
  state.playbackAckStartedAt = performance.now();
  state.playbackClock = state.audioContext ? state.audioContext.currentTime : 0;
  state.playerFrames = [];
  for (const node of state.playbackNodes) {
    try {
      node.stop();
    } catch (_err) {}
  }
  state.playbackNodes.clear();
  const responseId = state.activeResponseId;
  state.activeResponseId = null;
  setStatus(playbackState, "cleared");
  if (state.audioWs && state.audioWs.readyState === WebSocket.OPEN) {
    state.audioWs.send(JSON.stringify({
      type: "playback_stop_ack",
      response_id: responseId,
      elapsed_ms: Math.round(performance.now() - state.playbackAckStartedAt),
    }));
  }
}

async function shutdownAudioPipeline() {
  clearPlayback();
  if (state.playerTask !== null) {
    clearInterval(state.playerTask);
    state.playerTask = null;
  }
  if (state.workletNode) {
    state.workletNode.port.postMessage({ type: "stop" });
    state.workletNode.port.onmessage = null;
    state.workletNode.disconnect();
  }
  if (state.sourceNode) {
    state.sourceNode.disconnect();
  }
  if (state.mediaStream) {
    for (const track of state.mediaStream.getTracks()) {
      track.stop();
    }
  }
  if (state.audioContext) {
    await state.audioContext.close();
  }
  state.audioContext = null;
  state.mediaStream = null;
  state.sourceNode = null;
  state.workletNode = null;
  setStatus(micState, "off");
}

async function ensureAudioPipeline() {
  if (state.audioContext) {
    return;
  }
  state.mediaStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    },
  });
  state.audioContext = new AudioContext();
  await state.audioContext.audioWorklet.addModule(`/static/pcm-capture-processor.js?v=${state.assetVersion}`);
  state.browserSampleRate = state.audioContext.sampleRate;
  state.sourceNode = state.audioContext.createMediaStreamSource(state.mediaStream);
  state.workletNode = new AudioWorkletNode(state.audioContext, "pcm-capture-processor");
  state.workletNode.port.postMessage({
    type: "config",
    chunk_duration_ms: UPLINK_FRAME_MS,
    context_base_wall_ms: Date.now() - (state.audioContext.currentTime * 1000),
  });
  state.workletNode.port.onmessage = (event) => {
    const payload = event.data;
    if (!payload || payload.type !== "chunk") {
      return;
    }
    sendUplinkAudio(payload);
  };
  const silentGain = state.audioContext.createGain();
  silentGain.gain.value = 0;
  state.sourceNode.connect(state.workletNode);
  state.workletNode.connect(silentGain).connect(state.audioContext.destination);
  setStatus(micState, "on");
}

function sendUplinkAudio(payload) {
  if (!state.uplinkWs || state.uplinkWs.readyState !== WebSocket.OPEN || !state.started || !state.uplinkBound) {
    return;
  }
  const bufferedAmount = state.uplinkWs.bufferedAmount;
  if (bufferedAmount > MAX_UPLINK_BUFFER) {
    console.debug("uplink buffered", { bufferedAmount, seq: payload.seq });
    return;
  }
  const frame = packAudioFrame(
    {
      type: "audio_in",
      session_id: state.sessionId,
      seq: payload.seq,
      captured_at_ms: payload.captured_at_ms,
      sample_rate: payload.sample_rate || state.browserSampleRate,
      frame_ms: UPLINK_FRAME_MS,
      uplink_buffered_bytes: bufferedAmount,
    },
    payload.pcm_buffer,
  );
  state.uplinkWs.send(frame);
}

async function openUplinkSocket() {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(wsUrl("/ws/uplink"));
    ws.binaryType = "arraybuffer";
    state.uplinkWs = ws;
    ws.onopen = () => {
      if (!state.sessionId) {
        reject(new Error("missing sessionId"));
        return;
      }
      ws.send(JSON.stringify({ type: "bind", session_id: state.sessionId }));
      state.uplinkBound = true;
      resolve();
    };
    ws.onerror = () => reject(new Error("uplink websocket failed"));
    ws.onclose = () => {
      state.uplinkBound = false;
      state.uplinkWs = null;
    };
  });
}

async function openAudioSocket() {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(wsUrl("/ws/downlink-audio"));
    ws.binaryType = "arraybuffer";
    state.audioWs = ws;
    ws.onopen = () => {
      if (!state.sessionId) {
        reject(new Error("missing sessionId"));
        return;
      }
      ws.send(JSON.stringify({ type: "bind", session_id: state.sessionId }));
      state.audioBound = true;
      resolve();
    };
    ws.onmessage = (event) => {
      if (typeof event.data === "string") {
        return;
      }
      const { header, payload } = unpackAudioFrame(event.data);
      if (header.type === "tts_chunk") {
        queuePcmChunk(payload, header.sample_rate, header.response_id, header.generated_at_ms);
      }
    };
    ws.onerror = () => reject(new Error("downlink audio websocket failed"));
    ws.onclose = () => {
      state.audioBound = false;
      state.audioWs = null;
    };
  });
}

async function connect() {
  connectBtn.disabled = true;
  setStatus(socketState, "connecting");
  const ws = new WebSocket(wsUrl("/ws/control"));
  state.controlWs = ws;

  ws.onopen = () => {
    disconnectBtn.disabled = false;
    setStatus(socketState, "control-open");
  };

  ws.onmessage = async (event) => {
    const message = JSON.parse(event.data);
    if (message.type === "ready") {
      console.info("agent_voice_app frontend version", state.assetVersion);
      state.streamId = message.stream_id;
      state.sessionId = message.session_id;
      await ensureAudioPipeline();
      await state.audioContext.resume();
      await openUplinkSocket();
      await openAudioSocket();
      state.started = true;
      state.playbackClock = state.audioContext.currentTime;
      setStatus(socketState, "ready");
      ws.send(JSON.stringify({ type: "start", sample_rate: state.browserSampleRate }));
      state.workletNode.port.postMessage({ type: "start" });
      return;
    }
    if (message.type === "session_control") {
      if (message.event === "phase") {
        setStatus(phaseState, message.phase.toLowerCase());
      } else if (message.event === "turn_debug") {
        state.pendingDebugPayload = message.payload;
        scheduleRender();
      } else if (message.event === "metrics") {
        state.pendingMetricsPayload = message.payload;
        scheduleRender();
      } else if (message.event === "error") {
        setStatus(socketState, `error: ${message.message}`);
      }
      return;
    }
    if (message.type === "turn_event") {
      if (message.kind === "speech_start") {
        state.liveText.user = state.liveText.user || "...";
        scheduleRender();
      }
      return;
    }
    if (message.type === "interrupt") {
      if (message.kind === "clear_audio") {
        clearPlayback();
        state.liveText.assistant = "...";
        scheduleRender();
      }
      return;
    }
    if (message.type === "asr_partial") {
      state.liveText.user = message.text || "...";
      scheduleRender();
      return;
    }
    if (message.type === "asr_final") {
      state.liveText.user = message.text || "...";
      appendBubble("user", message.text || "");
      scheduleRender();
      return;
    }
    if (message.type === "llm_token") {
      state.liveText.assistant = `${state.liveText.assistant === "..." ? "" : state.liveText.assistant}${message.text || ""}` || "...";
      if (message.response_id) {
        state.activeResponseId = message.response_id;
      }
      scheduleRender();
      return;
    }
    if (message.type === "llm_final") {
      state.llmFinalText = message.text || "";
      state.liveText.assistant = state.llmFinalText || "...";
      appendBubble("assistant", state.llmFinalText);
      scheduleRender();
    }
  };

  ws.onclose = async () => {
    disconnectBtn.disabled = true;
    connectBtn.disabled = false;
    state.started = false;
    state.sessionId = null;
    state.streamId = null;
    setStatus(socketState, "closed");
    setStatus(phaseState, "offline");
    setStatus(playbackState, "idle");
    if (state.uplinkWs && state.uplinkWs.readyState === WebSocket.OPEN) {
      state.uplinkWs.close();
    }
    if (state.audioWs && state.audioWs.readyState === WebSocket.OPEN) {
      state.audioWs.close();
    }
    await shutdownAudioPipeline();
  };
}

async function disconnect() {
  if (state.controlWs && state.controlWs.readyState === WebSocket.OPEN) {
    state.controlWs.send(JSON.stringify({ type: "stop" }));
    state.controlWs.close();
  }
  if (state.uplinkWs && state.uplinkWs.readyState === WebSocket.OPEN) {
    state.uplinkWs.close();
  }
  if (state.audioWs && state.audioWs.readyState === WebSocket.OPEN) {
    state.audioWs.close();
  }
}

connectBtn.addEventListener("click", () => void connect());
disconnectBtn.addEventListener("click", () => void disconnect());
clearBtn.addEventListener("click", clearTranscript);
