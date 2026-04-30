# 图片标签识别技术方案

## 最新更新：多模型支持架构

### 架构概述
系统现已支持3种不同的AI模型进行元素识别，用户可根据性能需求和资源情况选择最适合的模型：

1. **Florence-2 base-ft** (默认，准确) - 高准确度，多任务能力强；推荐 GPU（16GB+）或高性能 CPU
2. **BLIP Image Captioning Large** (快速、通用) - 通用图像描述能力，适合快速候选生成；推荐 GPU（4-8GB）
3. **ViT-GPT2 Image Captioning** (轻量、稳定) - 轻量级图像描述，资源需求低，可在 CPU 上运行

### 技术实现

#### 1. 后端多模型支持
**核心文件**：`backend/app/services/image_recognition.py`

**关键改进**：
- **统一接口设计**：所有模型通过相同的 `ImageRecognitionService` 接口调用
- **智能模型检测**：自动识别 Florence 多任务模型、BLIP-2 模型、通用 captioning 模型
- **手动模型加载**：使用 `AutoProcessor` + `AutoModelForCausalLM`/`Blip2ForConditionalGeneration`/`Florence2ForConditionalGeneration` 手动加载，避免 `ImageToTextPipeline` 兼容性问题
- **管道缓存机制**：避免重复加载相同模型，提高性能
- **详细加载日志**：便于调试和监控模型加载状态

#### 2. 前端模型选择器
**核心文件**：`frontend/index.html`, `frontend/app.js`

**功能特性**：
- **动态模型列表**：从后端 API 获取最新模型配置
- **模型描述显示**：下拉框仅显示模型 label，description 在下方单独展示
- **实时预览**：选择模型时立即显示详细描述信息
- **用户友好界面**：中文标签和说明，便于理解

#### 3. 配置管理
**核心文件**：`backend/app/core/config.py`

**配置结构**：
```python
SUPPORTED_ELEMENT_RECOGNITION_MODELS = {
    "florence-community/Florence-2-base-ft": {
        "label": "Florence-2 base-ft（默认，准确）",
        "desc": "高准确度，多任务能力强；推荐 GPU（16GB+）或高性能 CPU。",
    },
    "Salesforce/blip-image-captioning-large": {
        "label": "BLIP 图像描述（快速、通用）",
        "desc": "通用图像描述能力，适合快速候选生成；推荐有 GPU（4-8GB）。",
    },
    "nlpconnect/vit-gpt2-image-captioning": {
        "label": "ViT-GPT2 图像描述（轻量、稳定）",
        "desc": "轻量级图像描述，资源需求低，可在 CPU 上运行但较慢。",
    },
}
```

## 识别流程

### 核心流程
```
用户上传图片并选择参数
    ↓
选择元素提取模型（3种可选）
    ↓
AI识别（根据选择的模型进行图像理解）
    ↓
获得英文标签列表
    ↓
使用 TranslationService 翻译（根据用户选择的语言）
    ↓
返回结果用于重命名
```

### 模型加载策略
- **Florence-2 模型**：使用 `Florence2ForConditionalGeneration` + `AutoProcessor`
- **BLIP 模型**：使用 `BlipForImageCaptioning` + `BlipProcessor`
- **ViT-GPT2 模型**：使用 `AutoProcessor` + `AutoModelForCausalLM`

### 翻译流程
```
英文标签输入
    ↓
第1级：精确匹配（大小写不敏感）
    ↓
第2级：复合词处理（下划线/连字符分解）
    ↓
第3级：模糊匹配（相似度算法）
    ↓
输出翻译结果或保留原英文
```

## 识别规则面板

前端"识别规则"面板实时显示以下信息：

| 项目 | 说明 |
|------|------|
| 当前元素提取模型 | 当前选中的AI模型名称 |
| 候选校验模型 | 用于验证候选标签的模型（CLIP） |
| 支持格式 | 支持的图片格式列表 |
| 命名规则 | 输出文件名格式：`camera_type_label1_label2.ext` |
| 运行设备 | 当前使用的计算设备（cuda/cpu） |

## 文件变更清单

1. **新增**：`backend/app/services/translation.py` - 翻译服务
2. **修改**：`backend/app/services/image_recognition.py` - 多模型支持、手动加载、错误处理
3. **修改**：`backend/app/core/config.py` - 添加3种支持模型配置
4. **修改**：`backend/app/routes/upload.py` - 使用新翻译服务
5. **修改**：`frontend/app.js` - 模型选择器优化，仅显示 label

## 部署指南
1. 安装依赖：`pip install -r backend/requirements.txt`
2. 配置环境变量（可选）：`HF_ENDPOINT`, `HUGGINGFACE_TOKEN`
3. 启动后端服务：`python backend/run.py`
4. 测试：上传图片，测试不同模型和语言设置