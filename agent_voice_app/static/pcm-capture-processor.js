class PcmCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.chunkSamples = Math.max(1, Math.round(sampleRate * 0.08));
    this.pending = [];
    this.contextBaseWallMs = 0;
    this.seq = 0;
    this.enabled = false;
    this.port.onmessage = (event) => {
      const payload = event.data || {};
      if (payload.type === "start") {
        this.enabled = true;
        this.pending = [];
        return;
      }
      if (payload.type === "stop") {
        this.enabled = false;
        this.pending = [];
        return;
      }
      if (payload.type !== "config") {
        return;
      }
      if (payload.chunk_duration_ms) {
        this.chunkSamples = Math.max(1, Math.round(sampleRate * (payload.chunk_duration_ms / 1000)));
      }
      if (payload.context_base_wall_ms) {
        this.contextBaseWallMs = payload.context_base_wall_ms;
      }
    };
  }

  process(inputs) {
    if (!this.enabled) {
      return true;
    }
    const input = inputs[0];
    if (!input || !input[0]) {
      return true;
    }
    const channel = input[0];
    for (let i = 0; i < channel.length; i += 1) {
      this.pending.push(channel[i]);
    }
    while (this.pending.length >= this.chunkSamples) {
      const chunk = new Float32Array(this.pending.splice(0, this.chunkSamples));
      this.seq += 1;
      this.port.postMessage({
        type: "chunk",
        samples: chunk,
        seq: this.seq,
        chunk_samples: this.chunkSamples,
        captured_at_ms: this.contextBaseWallMs + (currentTime * 1000),
        sample_rate: sampleRate,
      });
    }
    return true;
  }
}

registerProcessor("pcm-capture-processor", PcmCaptureProcessor);
