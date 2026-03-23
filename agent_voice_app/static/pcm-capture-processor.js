class PcmCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.chunkSamples = Math.max(1, Math.round(sampleRate * 0.16));
    this.pending = [];
  }

  process(inputs) {
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
      this.port.postMessage({
        samples: chunk,
        captured_at_ms: currentTime * 1000,
        sample_rate: sampleRate,
      });
    }
    return true;
  }
}

registerProcessor("pcm-capture-processor", PcmCaptureProcessor);
