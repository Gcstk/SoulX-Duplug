class PcmPlaybackProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.queue = [];
    this.current = null;
    this.currentOffset = 0;
    this.bufferedSamples = 0;
    this.maxBufferedSamples = Math.max(1, Math.round(sampleRate * 0.2));
    this.reportCounter = 0;
    this.port.onmessage = (event) => {
      const payload = event.data || {};
      if (payload.type === "config") {
        if (payload.max_buffer_ms) {
          this.maxBufferedSamples = Math.max(1, Math.round(sampleRate * (payload.max_buffer_ms / 1000)));
        }
        return;
      }
      if (payload.type === "clear") {
        this.queue = [];
        this.current = null;
        this.currentOffset = 0;
        this.bufferedSamples = 0;
        this._report();
        return;
      }
      if (payload.type !== "enqueue" || !payload.pcm_buffer) {
        return;
      }
      const sourceRate = payload.sample_rate || sampleRate;
      const input = new Int16Array(payload.pcm_buffer);
      const chunk = this._resampleToContext(input, sourceRate);
      if (!chunk.length) {
        return;
      }
      this.queue.push(chunk);
      this.bufferedSamples += chunk.length;
      while (this.bufferedSamples > this.maxBufferedSamples && this.queue.length > 1) {
        const dropped = this.queue.shift();
        this.bufferedSamples -= dropped.length;
      }
      this._report();
    };
  }

  _resampleToContext(int16Samples, sourceRate) {
    if (!int16Samples.length) {
      return new Float32Array(0);
    }
    if (sourceRate === sampleRate) {
      const sameRate = new Float32Array(int16Samples.length);
      for (let i = 0; i < int16Samples.length; i += 1) {
        sameRate[i] = int16Samples[i] / 0x7fff;
      }
      return sameRate;
    }
    const outputLength = Math.max(1, Math.round(int16Samples.length * sampleRate / sourceRate));
    const output = new Float32Array(outputLength);
    const ratio = sourceRate / sampleRate;
    for (let i = 0; i < outputLength; i += 1) {
      const srcIndex = i * ratio;
      const left = Math.floor(srcIndex);
      const right = Math.min(left + 1, int16Samples.length - 1);
      const weight = srcIndex - left;
      const leftValue = int16Samples[left] / 0x7fff;
      const rightValue = int16Samples[right] / 0x7fff;
      output[i] = leftValue + ((rightValue - leftValue) * weight);
    }
    return output;
  }

  _report() {
    this.port.postMessage({
      type: "buffer",
      buffered_ms: Math.round((this.bufferedSamples / sampleRate) * 1000),
      queued_chunks: this.queue.length + (this.current ? 1 : 0),
    });
  }

  process(_inputs, outputs) {
    const output = outputs[0];
    if (!output || !output[0]) {
      return true;
    }
    const channel = output[0];
    channel.fill(0);
    let writeIndex = 0;
    while (writeIndex < channel.length) {
      if (!this.current) {
        this.current = this.queue.shift() || null;
        this.currentOffset = 0;
        if (!this.current) {
          break;
        }
      }
      const remaining = this.current.length - this.currentOffset;
      const writable = Math.min(remaining, channel.length - writeIndex);
      channel.set(this.current.subarray(this.currentOffset, this.currentOffset + writable), writeIndex);
      this.currentOffset += writable;
      writeIndex += writable;
      this.bufferedSamples = Math.max(0, this.bufferedSamples - writable);
      if (this.currentOffset >= this.current.length) {
        this.current = null;
        this.currentOffset = 0;
      }
    }
    this.reportCounter += 1;
    if (this.reportCounter >= 32) {
      this.reportCounter = 0;
      this._report();
    }
    return true;
  }
}

registerProcessor("pcm-playback-processor", PcmPlaybackProcessor);
