# 声镜部署步骤

Streamlit Community Cloud 需要从 GitHub 仓库部署，不能直接部署本地文件夹。

## 1. 推到 GitHub

在 `voice-analyzer-mvp` 文件夹里初始化 Git，然后推到一个新的 GitHub 仓库。

```powershell
cd "C:\Users\aotly\Documents\Codex\2026-06-16\praat-f0-f4-spacing-ai-api\outputs\voice-analyzer-mvp"
git init
git add .
git commit -m "Initial VoiceScope MVP"
git branch -M main
git remote add origin https://github.com/你的用户名/voice-analyzer-mvp.git
git push -u origin main
```

## 2. 在 Streamlit Cloud 新建 App

选择你的 GitHub 仓库，入口文件填：

```text
app.py
```

## 3. 配置 Secrets

进入 Streamlit App 的 Settings -> Secrets，填入：

```toml
VOICESCOPE_PROVIDER = "OpenAI"
OPENAI_API_KEY = "你的 OpenAI Key"
OPENAI_MODEL = "gpt-5-mini"
```

也可以用 Gemini：

```toml
VOICESCOPE_PROVIDER = "Gemini"
GEMINI_API_KEY = "你的 Gemini Key"
GEMINI_MODEL = "gemini-2.5-flash"
```

不要把真实 Key 写进 GitHub。

## 4. 前台效果

用户不会看到模型选择、API Key 输入框或报告生成方式。后台没配置 Key 时，产品会自动使用本地规则报告。
