from __future__ import annotations

import exifread
import os
import shutil
from datetime import datetime
from enum import Enum
from pathlib import Path

from ..core.config import settings


class PhotoType(str, Enum):
    """照片类型枚举"""
    Portrait = "Portrait"
    EmotionalPortrait = "Emotional Portrait"
    BusinessPortrait = "Business Portrait"
    NaturalScenery = "Natural Scenery"
    StarrySky = "Starry Sky"
    SunriseSunset = "Sunrise & Sunset"
    MountainsWaters = "Mountains & Waters"
    ForestPlants = "Forest & Plants"
    IceSnowFog = "Ice, Snow & Fog"
    Birds = "Birds"
    WildAnimals = "Wild Animals"
    FlowersPlants = "Flowers & Plants"
    MacroInsects = "Macro Insects"
    CityStreetscape = "City Streetscape"
    Architecture = "Architecture"
    NightScene = "Night Scene"
    StillLife = "Still Life"
    FoodCuisine = "Food & Cuisine"
    DailySnapshots = "Daily Snapshots"
    TravelDocumentary = "Travel Documentary"


PHOTO_TYPE_TRANSLATIONS = {
    "Portrait": "人像",
    "Emotional Portrait": "情感人像",
    "Business Portrait": "商务人像",
    "Natural Scenery": "自然风景",
    "Starry Sky": "星空",
    "Sunrise & Sunset": "日出日落",
    "Mountains & Waters": "山水",
    "Forest & Plants": "森林植物",
    "Ice, Snow & Fog": "冰雪雾",
    "Birds": "鸟类",
    "Wild Animals": "野生动物",
    "Flowers & Plants": "花草",
    "Macro Insects": "微距昆虫",
    "City Streetscape": "城市街景",
    "Architecture": "建筑",
    "Night Scene": "夜景",
    "Still Life": "静物",
    "Food & Cuisine": "美食",
    "Daily Snapshots": "日常快照",
    "Travel Documentary": "旅行纪录",
}

LABEL_TRANSLATIONS = {
    "person": "人物",
    "face": "面孔",
    "portrait": "人像",
    "people": "人物",
    "group": "人群",
    "man": "男人",
    "woman": "女人",
    "child": "儿童",
    "baby": "婴儿",
    "girl": "女孩",
    "boy": "男孩",
    "suit": "西装",
    "business": "商务",
    "formal": "正式",
    "office": "办公室",
    "professional": "专业",
    "tie": "领带",
    "jacket": "外套",
    "smile": "微笑",
    "laugh": "大笑",
    "happy": "快乐",
    "sad": "悲伤",
    "cry": "哭泣",
    "emotion": "情感",
    "expression": "表情",
    "mountain": "山",
    "mountains": "山脉",
    "sky": "天空",
    "cloud": "云",
    "landscape": "风景",
    "scenery": "景色",
    "horizon": "地平线",
    "nature": "自然",
    "outdoor": "户外",
    "star": "星星",
    "stars": "星星",
    "starry": "星空",
    "night sky": "夜空",
    "milky way": "银河",
    "galaxy": "星系",
    "sunset": "日落",
    "sunrise": "日出",
    "dawn": "黎明",
    "dusk": "黄昏",
    "golden hour": "黄金时刻",
    "twilight": "暮光",
    "water": "水",
    "lake": "湖",
    "river": "河",
    "ocean": "海洋",
    "sea": "海",
    "beach": "海滩",
    "waterfall": "瀑布",
    "pond": "池塘",
    "snow": "雪",
    "ice": "冰",
    "fog": "雾",
    "mist": "薄雾",
    "frost": "霜",
    "winter": "冬天",
    "cold": "寒冷",
    "freezing": "结冰",
    "forest": "森林",
    "tree": "树",
    "trees": "树木",
    "woods": "树林",
    "jungle": "丛林",
    "green": "绿色",
    "plant": "植物",
    "plants": "植物",
    "bird": "鸟",
    "birds": "鸟类",
    "eagle": "鹰",
    "hawk": "鹰",
    "sparrow": "麻雀",
    "pigeon": "鸽子",
    "parrot": "鹦鹉",
    "owl": "猫头鹰",
    "animal": "动物",
    "wild": "野生",
    "wildlife": "野生动物",
    "lion": "狮子",
    "tiger": "老虎",
    "elephant": "大象",
    "deer": "鹿",
    "bear": "熊",
    "wolf": "狼",
    "fox": "狐狸",
    "insect": "昆虫",
    "butterfly": "蝴蝶",
    "bee": "蜜蜂",
    "ant": "蚂蚁",
    "beetle": "甲虫",
    "dragonfly": "蜻蜓",
    "macro": "微距",
    "close-up": "特写",
    "flower": "花",
    "flowers": "花朵",
    "blossom": "花朵",
    "bloom": "盛开",
    "petal": "花瓣",
    "rose": "玫瑰",
    "tulip": "郁金香",
    "orchid": "兰花",
    "city": "城市",
    "street": "街道",
    "skyline": "天际线",
    "urban": "城市",
    "downtown": "市中心",
    "building": "建筑",
    "buildings": "建筑物",
    "skyscraper": "摩天大楼",
    "architecture": "建筑",
    "church": "教堂",
    "temple": "寺庙",
    "monument": "纪念碑",
    "statue": "雕像",
    "bridge": "桥梁",
    "tower": "塔",
    "night": "夜晚",
    "nighttime": "夜间",
    "dark": "黑暗",
    "light": "灯光",
    "neon": "霓虹灯",
    "illuminated": "照明",
    "object": "物体",
    "objects": "物品",
    "product": "产品",
    "still life": "静物",
    "arrangement": "布置",
    "decorative": "装饰",
    "food": "食物",
    "meal": "餐",
    "dish": "菜肴",
    "cuisine": "美食",
    "cooking": "烹饪",
    "restaurant": "餐厅",
    "plate": "盘子",
    "bowl": "碗",
    "daily": "日常",
    "snapshot": "快照",
    "casual": "休闲",
    "everyday": "每天",
    "lifestyle": "生活方式",
    "home": "家",
    "indoor": "室内",
    "travel": "旅行",
    "trip": "旅行",
    "vacation": "假期",
    "tour": "旅游",
    "tourist": "游客",
    "landmark": "地标",
    "destination": "目的地",
    "dog": "狗",
    "cat": "猫",
    "horse": "马",
    "cow": "牛",
    "sheep": "羊",
    "fish": "鱼",
    "car": "汽车",
    "truck": "卡车",
    "bus": "公共汽车",
    "train": "火车",
    "bicycle": "自行车",
    "motorcycle": "摩托车",
    "airplane": "飞机",
    "boat": "船",
    "house": "房子",
    "bridge": "桥梁",
    "garden": "花园",
    "fruit": "水果",
    "cake": "蛋糕",
    "coffee": "咖啡",
    "drink": "饮料",
    "table": "桌子",
    "book": "书",
    "document": "文档",
    "screen": "屏幕",
    "laptop": "笔记本电脑",
    "computer": "电脑",
    "phone": "手机",
    "television": "电视",
    "keyboard": "键盘",
    "sofa": "沙发",
    "chair": "椅子",
    "bed": "床",
    "room": "房间",
    "kitchen": "厨房",
    "bathroom": "浴室",
    "artwork": "艺术品",
    "painting": "绘画",
    "poster": "海报",
    "logo": "标志",
    "text": "文字",
}


def get_camera_make(image_path: str) -> str | None:
    """读取图片的EXIF元数据，提取相机制造商。"""
    try:
        with open(image_path, 'rb') as f:
            tags = exifread.process_file(f, details=False)
        make = tags.get('Image Make')
        if make:
            return str(make.values).strip().replace(' ', '_')[:30]
        return None
    except Exception:
        return None


def classify_photo_type(labels: list[str], language: str = "en") -> str:
    """根据AI识别的标签判断照片类型，返回匹配度最高的类型。

    Args:
        labels: AI识别的标签列表
        language: 语言选项，'en' 为英文，'zh' 为中文

    Returns:
        照片类型字符串，始终返回一个类型
    """
    # 将标签转换为小写以便匹配
    label_set = {label.lower() for label in labels}
    
    # 计算匹配度的函数
    def count_matches(keywords: set[str]) -> int:
        """计算关键词集合在标签中的匹配数量"""
        return sum(1 for kw in keywords if any(kw in label for label in label_set))

    # 定义所有照片类型及其关键词
    type_keywords = {
        PhotoType.Portrait: {"person", "face", "portrait", "people", "man", "woman", "child", "baby", "girl", "boy"},
        PhotoType.BusinessPortrait: {"suit", "business", "formal", "office", "professional", "tie", "jacket"},
        PhotoType.EmotionalPortrait: {"smile", "laugh", "happy", "sad", "cry", "emotion", "expression"},
        PhotoType.NaturalScenery: {"mountain", "mountains", "sky", "cloud", "landscape", "scenery", "horizon", "nature", "outdoor"},
        PhotoType.StarrySky: {"star", "stars", "starry", "night sky", "milky way", "galaxy"},
        PhotoType.SunriseSunset: {"sunset", "sunrise", "dawn", "dusk", "golden hour", "twilight"},
        PhotoType.MountainsWaters: {"water", "lake", "river", "ocean", "sea", "beach", "waterfall", "pond"},
        PhotoType.IceSnowFog: {"snow", "ice", "fog", "mist", "frost", "winter", "cold", "freezing"},
        PhotoType.ForestPlants: {"forest", "tree", "trees", "woods", "jungle", "green", "plant", "plants"},
        PhotoType.Birds: {"bird", "birds", "eagle", "hawk", "sparrow", "pigeon", "parrot", "owl"},
        PhotoType.WildAnimals: {"animal", "wild", "wildlife", "lion", "tiger", "elephant", "deer", "bear", "wolf", "fox"},
        PhotoType.MacroInsects: {"insect", "butterfly", "bee", "ant", "beetle", "dragonfly", "macro", "close-up"},
        PhotoType.FlowersPlants: {"flower", "flowers", "blossom", "bloom", "petal", "rose", "tulip", "orchid"},
        PhotoType.CityStreetscape: {"city", "street", "urban", "downtown", "building", "buildings", "skyscraper"},
        PhotoType.Architecture: {"architecture", "church", "temple", "monument", "statue", "bridge", "tower"},
        PhotoType.NightScene: {"night", "nighttime", "dark", "light", "neon", "illuminated"},
        PhotoType.StillLife: {"object", "objects", "product", "still life", "arrangement", "decorative"},
        PhotoType.FoodCuisine: {"food", "meal", "dish", "cuisine", "cooking", "restaurant", "plate", "bowl"},
        PhotoType.DailySnapshots: {"daily", "snapshot", "casual", "everyday", "lifestyle", "home", "indoor"},
        PhotoType.TravelDocumentary: {"travel", "trip", "vacation", "tour", "tourist", "landmark", "destination"},
    }
    
    # 特殊组合类型需要额外判断
    # MountainsWaters 需要同时匹配 scenery 和 water
    scenery_count = count_matches(type_keywords[PhotoType.NaturalScenery])
    water_count = count_matches(type_keywords[PhotoType.MountainsWaters])
    if scenery_count > 0 and water_count > 0:
        mountains_waters_score = scenery_count + water_count
    else:
        mountains_waters_score = 0
    
    # NightScene 需要同时匹配 city 和 night
    city_count = count_matches(type_keywords[PhotoType.CityStreetscape])
    night_count = count_matches(type_keywords[PhotoType.NightScene])
    if city_count > 0 and night_count > 0:
        night_scene_score = city_count + night_count
    else:
        night_scene_score = 0
    
    # 计算所有类型的匹配度
    scores = {}
    for photo_type, keywords in type_keywords.items():
        if photo_type in [PhotoType.MountainsWaters, PhotoType.NightScene]:
            continue  # 特殊类型单独处理
        scores[photo_type] = count_matches(keywords)
    
    # 添加特殊类型的分数
    if mountains_waters_score > 0:
        scores[PhotoType.MountainsWaters] = mountains_waters_score
    if night_scene_score > 0:
        scores[PhotoType.NightScene] = night_scene_score
    
    # 如果没有匹配，返回默认类型 "Daily Snapshots"
    if not scores or max(scores.values()) == 0:
        best_type = PhotoType.DailySnapshots
    else:
        # 返回匹配度最高的类型
        best_type = max(scores.items(), key=lambda x: x[1])[0]
    
    # 根据语言返回对应的翻译
    if language == "zh":
        return PHOTO_TYPE_TRANSLATIONS.get(best_type.value, best_type.value)
    return best_type.value


def get_file_extension(filename: str) -> str:
    """Extract the lowercase extension without a leading dot."""
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


def is_allowed_file(filename: str) -> bool:
    """Check whether a file extension is supported."""
    return get_file_extension(filename) in settings.ALLOWED_EXTENSIONS


def sanitize_label(label: str) -> str:
    """Normalize a label so it is safe to use in a filename."""
    cleaned = "".join(char.lower() if char.isalnum() else "_" for char in label.strip())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")[:50]


def generate_output_filename(labels: list[str], extension: str) -> str:
    """Generate the renamed file according to aa_bb_cc style."""
    normalized = [sanitize_label(label) for label in labels[: settings.MAX_LABELS]]
    normalized = [label for label in normalized if label]
    base = "_".join(normalized)
    if not base:
        base = f"photo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return f"{base}.{extension}"


def ensure_directory(path: str) -> str:
    """Create a directory when needed and return its absolute path."""
    normalized = os.path.abspath(os.path.expanduser(path.strip()))
    os.makedirs(normalized, exist_ok=True)
    return normalized


def iter_image_files(source_dir: str, recursive: bool = True) -> list[str]:
    """Collect all supported image files from a source directory."""
    root = Path(os.path.abspath(os.path.expanduser(source_dir.strip())))
    if not root.exists():
        raise FileNotFoundError(f"Source directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Source path is not a directory: {root}")

    pattern = "**/*" if recursive else "*"
    return sorted(
        str(path)
        for path in root.glob(pattern)
        if path.is_file() and is_allowed_file(path.name)
    )


def build_unique_output_path(output_dir: str, labels: list[str], extension: str) -> str:
    """Build a unique output file path in the output directory."""
    output_root = Path(output_dir)
    candidate = output_root / generate_output_filename(labels, extension)
    return build_unique_path_for_name(output_root, candidate.name)


def build_unique_path_for_name(output_dir: str | Path, filename: str) -> str:
    """Build a unique output path for an explicit filename."""
    output_root = Path(output_dir)
    candidate = output_root / filename
    stem = candidate.stem
    suffix = candidate.suffix
    counter = 1

    while candidate.exists():
        candidate = output_root / f"{stem}_{counter}{suffix}"
        counter += 1

    return str(candidate)


def copy_to_output(source_path: str, output_path: str) -> None:
    """Copy a file to the output location."""
    shutil.copy2(source_path, output_path)
