# 声镜 VoiceScope MVP 使用说明

这是一个最小可用版本，也就是先把核心流程跑通：

上传声音 -> 声学分析 -> 图表 -> 中文声音画像报告。

## 你要打开哪个文件？

不用一个个看代码。你只需要记住：

- `app.py`：网页入口，运行它就能打开产品页面。
- `voice_core.py`：声音分析核心，负责 F0、F1-F4、formant spacing、HNR。
- `ai_reports.py`：AI 报告核心，负责 OpenAI / Gemini / 本地规则报告。
- `requirements.txt`：依赖清单，第一次运行前安装它。
- `assets/visual-direction.png`：我用 image 生成的审美参考图。

## 第一次怎么安装？

在这个文件夹里打开 PowerShell：

```powershell
cd "C:\Users\aotly\Documents\Codex\2026-06-16\praat-f0-f4-spacing-ai-api\outputs\voice-analyzer-mvp"
python -m pip install -r requirements.txt
```

注意：这里用的是 `praat-parselmouth`，不是 `parselmouth`。

## 怎么启动本地服务器？

在同一个文件夹运行：

```powershell
python -m streamlit run app.py --server.port 8501
```

然后浏览器打开：

```text
http://localhost:8501
```

我刚才就是用这个逻辑启动的，只是我让它在后台静默运行：

```powershell
Start-Process -WindowStyle Hidden -FilePath python -ArgumentList @('-m','streamlit','run','app.py','--server.port','8501','--server.headless','true') -WorkingDirectory 'C:\Users\aotly\Documents\Codex\2026-06-16\praat-f0-f4-spacing-ai-api\outputs\voice-analyzer-mvp'
```

你平时不用记这么长，直接用前面的 `python -m streamlit run app.py --server.port 8501` 就好。

## 怎么使用页面？

1. 打开 `http://localhost:8501`
2. 上传一段 `wav/flac/aiff/aif` 音频。
3. 左边填目标声线，比如“更自然女声”“更少年感”“更轻薄”。
4. 报告生成方式先选“本地规则”。
5. 如果你有 API Key，再切换成 OpenAI 或 Gemini。

## OpenAI API 怎么用？

页面左侧选择 `OpenAI`，填入你的 OpenAI API Key。

也可以在 PowerShell 里临时设置：

```powershell
$env:OPENAI_API_KEY="你的 key"
$env:OPENAI_MODEL="gpt-5-mini"
python -m streamlit run app.py --server.port 8501
```

## Gemini API 怎么用？

页面左侧选择 `Gemini`，填入你的 Gemini API Key。

也可以在 PowerShell 里临时设置：

```powershell
$env:GEMINI_API_KEY="你的 key"
$env:GEMINI_MODEL="gemini-2.5-flash"
python -m streamlit run app.py --server.port 8501
```

## 当前能力

- 上传 wav/flac/aiff/aif
- 提取平均 F0、F0 范围、F1-F4、formant spacing、HNR
- 显示主标签、副标签、声音 DNA、风格归类和成长方向
- 显示少女感、少年感、轻薄感、清亮感、气泡感、MTF 训练友好度、模仿可塑性等概念评分
- 把专业物理参数折叠到“Praat 党”区域
- 生成中文声音人格报告和训练建议
- 可选 OpenAI 或 Gemini 生成更细报告

## 当前边界

当前报告只提供声学倾向和训练反馈，不用于医学诊断、生理性别判断或身份判定。
