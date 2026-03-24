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
  assetVersion: "20260324b",
  controlWs: null,
  uplinkWs: null,
  streamId: null,
  sessionId: null,
  audioContext: null,
  mediaStream: null,
  sourceNode: null,
  workletNode: null,
  browserSampleRate: 16000,
  playbackClock: 0,
  playbackNodes: new Set(),
  activeResponseId: null,
  started: false,
  uplinkBound: false,
};

function controlWsUrl() {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}/ws/control`;
}

function uplinkWsUrl() {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}/ws/uplink`;
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

function resetLiveCard(target) {
  target.textContent = "...";
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
  resetLiveCard(userLive);
  resetLiveCard(assistantLive);
  debugState.textContent = "waiting for /turn events...";
  metricsState.textContent = "waiting for response metrics...";
}

function float32ToBase64Pcm16(floatBuffer) {
  const pcm = new Int16Array(floatBuffer.length);
  for (let i = 0; i < floatBuffer.length; i += 1) {
    const sample = Math.max(-1, Math.min(1, floatBuffer[i]));
    pcm[i] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
  }
  const bytes = new Uint8Array(pcm.buffer);
  let binary = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}

function base64ToInt16(base64) {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return new Int16Array(bytes.buffer);
}

function playPcmChunk(base64, sampleRate, responseId) {
  if (!state.audioContext) {
    return;
  }
  if (state.activeResponseId && responseId && responseId !== state.activeResponseId) {
    return;
  }
  if (responseId) {
    state.activeResponseId = responseId;
  }
  const samples = base64ToInt16(base64);
  const buffer = state.audioContext.createBuffer(1, samples.length, sampleRate);
  const channel = buffer.getChannelData(0);
  for (let i = 0; i < samples.length; i += 1) {
    channel[i] = samples[i] / 0x7fff;
  }
  const source = state.audioContext.createBufferSource();
  source.buffer = buffer;
  source.connect(state.audioContext.destination);
  const now = state.audioContext.currentTime;
  state.playbackClock = Math.max(state.playbackClock, now);
  source.start(state.playbackClock);
  state.playbackClock += buffer.duration;
  source.onended = () => state.playbackNodes.delete(source);
  state.playbackNodes.add(source);
  setStatus(playbackState, "streaming");
}

function clearPlayback() {
  state.playbackClock = state.audioContext ? state.audioContext.currentTime : 0;
  for (const node of state.playbackNodes) {
    try {
      node.stop();
    } catch (_err) {}
  }
  state.playbackNodes.clear();
  state.activeResponseId = null;
  setStatus(playbackState, "cleared");
}

async function shutdownAudioPipeline() {
  clearPlayback();
  if (state.workletNode) {
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
    chunk_duration_ms: 80,
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
  const wsSendStartedAt = performance.now();
  state.uplinkWs.send(JSON.stringify({
    type: "audio",
    encoding: "pcm16",
    seq: payload.seq,
    chunk_samples: payload.chunk_samples,
    sample_rate: payload.sample_rate || state.browserSampleRate,
    captured_at_ms: payload.captured_at_ms,
    audio_b64: float32ToBase64Pcm16(payload.samples),
  }));
  const captureToSendMs = performance.now() - wsSendStartedAt;
  const bufferedAmount = state.uplinkWs.bufferedAmount;
  if (bufferedAmount > 65536) {
    console.debug("uplink buffered", { bufferedAmount, seq: payload.seq, captureToSendMs });
  }
}

async function openUplinkSocket() {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(uplinkWsUrl());
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

async function connect() {
  connectBtn.disabled = true;
  setStatus(socketState, "connecting");
  const ws = new WebSocket(controlWsUrl());
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
      state.started = true;
      state.playbackClock = state.audioContext.currentTime;
      setStatus(socketState, "control+uplink");
      ws.send(JSON.stringify({ type: "start", sample_rate: state.browserSampleRate }));
      return;
    }
    if (message.type === "phase") {
      setStatus(phaseState, message.phase.toLowerCase());
      return;
    }
    if (message.type === "transcript") {
      const liveTarget = message.speaker === "user" ? userLive : assistantLive;
      liveTarget.textContent = message.text || "...";
      if (message.final) {
        appendBubble(message.speaker, message.text);
      }
      if (message.speaker === "assistant" && message.response_id) {
        state.activeResponseId = message.response_id;
      }
      return;
    }
    if (message.type === "audio") {
      playPcmChunk(message.audio_b64, message.sample_rate, message.response_id);
      return;
    }
    if (message.type === "clear_audio") {
      clearPlayback();
      assistantLive.textContent = "...";
      return;
    }
    if (message.type === "turn_debug") {
      debugState.textContent = JSON.stringify(message.payload, null, 2);
      return;
    }
    if (message.type === "metrics") {
      metricsState.textContent = JSON.stringify(message.payload, null, 2);
      return;
    }
    if (message.type === "error") {
      setStatus(socketState, `error: ${message.message}`);
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
}

connectBtn.addEventListener("click", () => void connect());
disconnectBtn.addEventListener("click", () => void disconnect());
clearBtn.addEventListener("click", clearTranscript);
