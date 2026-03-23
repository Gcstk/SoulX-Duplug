# SoulX Agent Voice App

独立浏览器语音 Agent 子项目。

## 能力

- 浏览器麦克风实时上行
- 通过根项目 `SoulX-Duplug /turn` 获取 `vad + asr + turn detecting`
- `speak` 后启动 `LLM -> Qwen TTS -> 浏览器播放`
- assistant 播报期间用户再次开口，立即打断并清空旧播放
- 所有 assistant 下行消息带 `response_id`，旧回调自动丢弃

## 交互说明

- 浏览器模式默认开启 `echoCancellation / noiseSuppression / autoGainControl`，优先保证本地全双工场景下的打断能力。
- 如果使用外放，浏览器和系统的回声消除质量会直接影响 barge-in 效果。
- 最稳定的测试方式仍然是耳机模式；否则 assistant 播放音可能回灌进麦克风，影响 `SoulX-Duplug` 的 turn 判断。

## 目录

```text
agent_voice_app/
  main.py
  server.py
  requirements.txt
  .env.example
  app/
  static/
  tests/
```

## 启动

先启动根目录 turn service：

```bash
cd /Users/geotk/workspace/audio/SoulX-Duplug
bash run.sh
```

再启动 agent app：

```bash
cd /Users/geotk/workspace/audio/SoulX-Duplug/agent_voice_app
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python main.py
```

浏览器打开：

```text
http://127.0.0.1:3040/web
```

## 配置

- `DUPLUG_WS_URL`: 根项目 `/turn` 地址
- `LLM_*`: OpenAI-compatible LLM
- `DASHSCOPE_API_KEY`: Qwen 实时 TTS key
- `AGENT_LOG_LEVEL`: 日志级别，默认 `INFO`
- `AGENT_LOG_FILE`: 日志文件路径，默认 `agent_voice_app/logs/agent_voice_app.log`

## 日志

- 控制台和文件都会输出统一日志
- 时间精确到毫秒
- 关键链路包含浏览器收包、Duplug 出入站、turn state、LLM 首 token、TTS 首音频、response 完成时延

## 测试

```bash
cd /Users/geotk/workspace/audio/SoulX-Duplug/agent_voice_app
python -m pytest tests -v
```
