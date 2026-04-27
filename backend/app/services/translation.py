"""Translation service for converting English labels to Chinese."""

from __future__ import annotations

import logging
from typing import Optional
import difflib

logger = logging.getLogger(__name__)

# 扩展的英文到中文的翻译字典
# 包含单词、短语和复合词
EXTENDED_LABEL_TRANSLATIONS = {
    # 人物相关
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
    "adult": "成人",
    "elder": "老人",
    "elderly": "老年人",
    "teenager": "青少年",
    "youth": "青年",
    "couple": "情侣",
    "family": "家庭",
    "sibling": "兄弟姐妹",
    "mother": "母亲",
    "father": "父亲",
    "parent": "父母",
    "children": "儿童",
    "human": "人类",
    "figure": "人物",
    
    # 服装和商务相关
    "suit": "西装",
    "business": "商务",
    "formal": "正式",
    "office": "办公室",
    "professional": "专业",
    "tie": "领带",
    "jacket": "外套",
    "shirt": "衬衫",
    "pants": "裤子",
    "dress": "连衣裙",
    "skirt": "裙子",
    "coat": "外衣",
    "hat": "帽子",
    "shoes": "鞋子",
    "shoe": "鞋",
    "sweater": "毛衣",
    "uniform": "制服",
    "costume": "服装",
    "clothing": "衣服",
    "wear": "服装",
    "apparel": "服装",
    "garment": "服装",
    
    # 表情和情感相关
    "smile": "微笑",
    "laugh": "大笑",
    "happy": "快乐",
    "sad": "悲伤",
    "cry": "哭泣",
    "emotion": "情感",
    "expression": "表情",
    "cheerful": "欢乐",
    "joyful": "欣喜",
    "gloomy": "阴沉",
    "depressed": "沮丧",
    "angry": "愤怒",
    "surprised": "惊讶",
    "confused": "困惑",
    "confused": "困惑",
    "neutral": "中立",
    "serious": "严肃",
    "relaxed": "放松",
    
    # 自然风景
    "mountain": "山",
    "mountains": "山脉",
    "sky": "天空",
    "cloud": "云",
    "clouds": "云彩",
    "landscape": "风景",
    "scenery": "景色",
    "scene": "场景",
    "horizon": "地平线",
    "nature": "自然",
    "outdoor": "户外",
    "outdoors": "户外",
    "vista": "景观",
    "panorama": "全景",
    "terrain": "地形",
    "valley": "山谷",
    "plateau": "高原",
    "hill": "山丘",
    "hills": "山丘",
    
    # 星空相关
    "star": "星星",
    "stars": "星星",
    "starry": "星空",
    "night sky": "夜空",
    "milky way": "银河",
    "galaxy": "星系",
    "constellation": "星座",
    "astronomy": "天文",
    "space": "太空",
    "cosmos": "宇宙",
    
    # 日出日落相关
    "sunset": "日落",
    "sunrise": "日出",
    "dawn": "黎明",
    "dusk": "黄昏",
    "golden hour": "黄金时刻",
    "twilight": "暮光",
    "sunlight": "阳光",
    "afternoon": "下午",
    "evening": "傍晚",
    "morning": "早晨",
    "daylight": "日光",
    
    # 水相关
    "water": "水",
    "lake": "湖",
    "river": "河",
    "ocean": "海洋",
    "sea": "海",
    "beach": "海滩",
    "waterfall": "瀑布",
    "pond": "池塘",
    "stream": "小溪",
    "creek": "溪流",
    "bay": "海湾",
    "gulf": "海湾",
    "inlet": "小湾",
    "strait": "海峡",
    "rapids": "急流",
    "waves": "波浪",
    "wave": "波浪",
    "splash": "溅起",
    "underwater": "水下",
    "aquatic": "水生的",
    
    # 冰雪雾相关
    "snow": "雪",
    "ice": "冰",
    "fog": "雾",
    "mist": "薄雾",
    "frost": "霜",
    "winter": "冬天",
    "cold": "寒冷",
    "freezing": "结冰",
    "frozen": "冻结",
    "blizzard": "暴风雪",
    "snowstorm": "暴风雪",
    "snowflake": "雪花",
    "icicle": "冰柱",
    "sleet": "冻雨",
    
    # 森林植物相关
    "forest": "森林",
    "tree": "树",
    "trees": "树木",
    "woods": "树林",
    "jungle": "丛林",
    "rainforest": "雨林",
    "green": "绿色",
    "plant": "植物",
    "plants": "植物",
    "vegetation": "植被",
    "foliage": "叶子",
    "leaf": "叶子",
    "leaves": "叶子",
    "branch": "树枝",
    "branches": "树枝",
    "trunk": "树干",
    "root": "根部",
    "bark": "树皮",
    "forest floor": "森林地面",
    "canopy": "树冠",
    "undergrowth": "灌木丛",
    
    # 鸟类相关
    "bird": "鸟",
    "birds": "鸟类",
    "eagle": "鹰",
    "hawk": "鹰",
    "sparrow": "麻雀",
    "pigeon": "鸽子",
    "parrot": "鹦鹉",
    "owl": "猫头鹰",
    "crow": "乌鸦",
    "dove": "鸽子",
    "swan": "天鹅",
    "goose": "鹅",
    "duck": "鸭子",
    "heron": "鹭",
    "crane": "鹤",
    "penguin": "企鹅",
    "flamingo": "火烈鸟",
    "peacock": "孔雀",
    "wings": "翅膀",
    "wing": "翅膀",
    "feather": "羽毛",
    "beak": "喙",
    
    # 野生动物相关
    "animal": "动物",
    "wildlife": "野生动物",
    "wild": "野生",
    "beast": "野兽",
    "creature": "生物",
    "mammal": "哺乳动物",
    "lion": "狮子",
    "tiger": "老虎",
    "elephant": "大象",
    "deer": "鹿",
    "bear": "熊",
    "wolf": "狼",
    "fox": "狐狸",
    "zebra": "斑马",
    "giraffe": "长颈鹿",
    "rhino": "犀牛",
    "hippo": "河马",
    "cheetah": "猎豹",
    "leopard": "豹",
    "panda": "熊猫",
    "koala": "考拉",
    "kangaroo": "袋鼠",
    "monkey": "猴子",
    "ape": "猿",
    "primate": "灵长类",
    
    # 昆虫和微距相关
    "insect": "昆虫",
    "butterfly": "蝴蝶",
    "bee": "蜜蜂",
    "ant": "蚂蚁",
    "beetle": "甲虫",
    "dragonfly": "蜻蜓",
    "moth": "飞蛾",
    "ladybug": "瓢虫",
    "caterpillar": "毛毛虫",
    "spider": "蜘蛛",
    "mosquito": "蚊子",
    "fly": "苍蝇",
    "grasshopper": "蚱蜢",
    "cricket": "蟋蟀",
    "macro": "微距",
    "close-up": "特写",
    "close up": "特写",
    "detail": "细节",
    "microscopic": "微观",
    
    # 花草相关
    "flower": "花",
    "flowers": "花朵",
    "blossom": "花朵",
    "bloom": "盛开",
    "petal": "花瓣",
    "petals": "花瓣",
    "rose": "玫瑰",
    "tulip": "郁金香",
    "orchid": "兰花",
    "lily": "百合",
    "sunflower": "向日葵",
    "daisy": "雏菊",
    "iris": "鸢尾花",
    "lotus": "莲花",
    "peony": "牡丹",
    "chrysanthemum": "菊花",
    "carnation": "康乃馨",
    "magnolia": "玉兰",
    "grass": "草",
    "grassland": "草地",
    "meadow": "草地",
    "herb": "草本植物",
    "moss": "苔藓",
    "fern": "蕨类",
    "vine": "藤蔓",
    "succulent": "多肉植物",
    
    # 城市街景相关
    "city": "城市",
    "street": "街道",
    "urban": "城市",
    "downtown": "市中心",
    "cityscape": "城市风景",
    "streetscape": "街道风景",
    "district": "地区",
    "neighborhood": "邻近区域",
    "suburb": "郊区",
    "metropolis": "大城市",
    "town": "小镇",
    "alley": "小巷",
    "lane": "小巷",
    "avenue": "大道",
    "boulevard": "林荫道",
    "plaza": "广场",
    "square": "广场",
    "marketplace": "市场",
    "sidewalk": "人行道",
    
    # 建筑相关
    "building": "建筑",
    "buildings": "建筑物",
    "architecture": "建筑",
    "skyscraper": "摩天大楼",
    "highrise": "高楼",
    "structure": "建筑",
    "edifice": "建筑物",
    "church": "教堂",
    "temple": "寺庙",
    "mosque": "清真寺",
    "synagogue": "犹太教堂",
    "monastery": "修道院",
    "convent": "女修道院",
    "monument": "纪念碑",
    "memorial": "纪念物",
    "statue": "雕像",
    "sculpture": "雕塑",
    "bridge": "桥梁",
    "tower": "塔",
    "pagoda": "塔",
    "castle": "城堡",
    "palace": "宫殿",
    "mansion": "豪宅",
    "villa": "别墅",
    "cottage": "小屋",
    "barn": "谷仓",
    "warehouse": "仓库",
    "factory": "工厂",
    "mill": "磨坊",
    "arch": "拱门",
    "dome": "穹顶",
    "roof": "屋顶",
    "wall": "墙",
    "window": "窗户",
    "door": "门",
    "gate": "门",
    "entrance": "入口",
    "column": "柱子",
    
    # 夜景相关
    "night": "夜晚",
    "nighttime": "夜间",
    "dark": "黑暗",
    "light": "灯光",
    "neon": "霓虹灯",
    "illuminated": "照明",
    "glow": "发光",
    "glowing": "发光",
    "gleam": "闪烁",
    "sparkle": "闪闪发光",
    "luminous": "发光的",
    "lantern": "灯笼",
    "lanterns": "灯笼",
    "candle": "蜡烛",
    "lamp": "灯",
    "streetlight": "路灯",
    "moonlight": "月光",
    "starlight": "星光",
    "shadow": "影子",
    "silhouette": "轮廓",
    "contrast": "对比",
    
    # 静物相关
    "object": "物体",
    "objects": "物品",
    "product": "产品",
    "still life": "静物",
    "still-life": "静物",
    "arrangement": "布置",
    "composition": "组合",
    "decorative": "装饰",
    "ornament": "装饰品",
    "artifact": "工艺品",
    "collection": "收集",
    "assemblage": "组合",
    
    # 美食相关
    "food": "食物",
    "meal": "餐",
    "dish": "菜肴",
    "cuisine": "美食",
    "cooking": "烹饪",
    "restaurant": "餐厅",
    "plate": "盘子",
    "bowl": "碗",
    "soup": "汤",
    "noodles": "面条",
    "rice": "米饭",
    "bread": "面包",
    "meat": "肉",
    "poultry": "家禽",
    "seafood": "海鲜",
    "vegetable": "蔬菜",
    "salad": "沙拉",
    "pasta": "意大利面",
    "pizza": "披萨",
    "hamburger": "汉堡",
    "sandwich": "三明治",
    "sushi": "寿司",
    "dumpling": "饺子",
    "dessert": "甜点",
    "pastry": "糕点",
    "pie": "馅饼",
    "cake": "蛋糕",
    "cupcake": "纸杯蛋糕",
    "cookie": "饼干",
    "biscuit": "饼干",
    "chocolate": "巧克力",
    "candy": "糖果",
    "fruit": "水果",
    "vegetable": "蔬菜",
    "apple": "苹果",
    "orange": "橙子",
    "banana": "香蕉",
    "strawberry": "草莓",
    "berry": "浆果",
    "grape": "葡萄",
    "watermelon": "西瓜",
    "coffee": "咖啡",
    "tea": "茶",
    "drink": "饮料",
    "beverage": "饮料",
    "alcohol": "酒精饮料",
    "wine": "葡萄酒",
    "beer": "啤酒",
    "cocktail": "鸡尾酒",
    "juice": "果汁",
    "smoothie": "冰沙",
    "milk": "牛奶",
    "utensil": "餐具",
    "silverware": "银制品",
    "cutlery": "刀叉",
    "napkin": "餐巾",
    "tablecloth": "桌布",
    
    # 日常快照相关
    "daily": "日常",
    "snapshot": "快照",
    "casual": "休闲",
    "everyday": "每天",
    "lifestyle": "生活方式",
    "home": "家",
    "indoor": "室内",
    "inside": "室内",
    "rooms": "房间",
    "domestic": "家庭",
    "housework": "家务",
    "chore": "杂务",
    "activity": "活动",
    "moment": "时刻",
    "candid": "坦诚",
    "spontaneous": "自发的",
    
    # 旅行相关
    "travel": "旅行",
    "trip": "旅行",
    "vacation": "假期",
    "holiday": "假期",
    "tour": "旅游",
    "tourist": "游客",
    "traveler": "旅行者",
    "journey": "旅程",
    "adventure": "冒险",
    "explorer": "探险家",
    "landmark": "地标",
    "landmark": "地标",
    "destination": "目的地",
    "souvenir": "纪念品",
    "passport": "护照",
    "luggage": "行李",
    "suitcase": "行李箱",
    "backpack": "背包",
    "map": "地图",
    "compass": "指南针",
    "trail": "小径",
    "campfire": "篝火",
    "campsite": "营地",
    "tent": "帐篷",
    "caravan": "旅行队",
    "vehicle": "车辆",
    
    # 宠物相关
    "dog": "狗",
    "dogs": "狗",
    "cat": "猫",
    "cats": "猫",
    "puppy": "小狗",
    "kitten": "小猫",
    "pet": "宠物",
    "paw": "爪子",
    "tail": "尾巴",
    "fur": "毛皮",
    "whisker": "胡须",
    "nose": "鼻子",
    "ear": "耳朵",
    "eyes": "眼睛",
    
    # 交通工具相关
    "car": "汽车",
    "automobile": "汽车",
    "vehicle": "车辆",
    "truck": "卡车",
    "van": "货车",
    "bus": "公共汽车",
    "coach": "客车",
    "train": "火车",
    "locomotive": "火车头",
    "railway": "铁路",
    "bicycle": "自行车",
    "bike": "自行车",
    "motorcycle": "摩托车",
    "scooter": "踏板车",
    "airplane": "飞机",
    "aircraft": "飞机",
    "helicopter": "直升机",
    "boat": "船",
    "ship": "船",
    "vessel": "船舶",
    "yacht": "游艇",
    "sailboat": "帆船",
    "canoe": "独木舟",
    "kayak": "皮划艇",
    "ferry": "渡轮",
    "cruise": "游轮",
    "subway": "地铁",
    "metro": "地铁",
    "tram": "电车",
    "trolley": "电车",
    "taxi": "出租车",
    "cab": "出租车",
    "road": "道路",
    "highway": "高速公路",
    "freeway": "高速公路",
    "intersection": "十字路口",
    "traffic": "交通",
    
    # 家具和室内相关
    "furniture": "家具",
    "table": "桌子",
    "sofa": "沙发",
    "couch": "沙发",
    "chair": "椅子",
    "bed": "床",
    "bedroom": "卧室",
    "living room": "客厅",
    "room": "房间",
    "rooms": "房间",
    "kitchen": "厨房",
    "bathroom": "浴室",
    "dining room": "餐厅",
    "hallway": "走廊",
    "corridor": "走廊",
    "staircase": "楼梯",
    "stairs": "楼梯",
    "elevator": "电梯",
    "door": "门",
    "window": "窗户",
    "curtain": "窗帘",
    "rug": "地毯",
    "carpet": "地毯",
    "painting": "绘画",
    "picture": "图片",
    "mirror": "镜子",
    "lamp": "灯",
    "shelf": "架子",
    "bookcase": "书架",
    "cabinet": "橱柜",
    "drawer": "抽屉",
    
    # 技术产品相关
    "technology": "技术",
    "tech": "科技",
    "computer": "电脑",
    "laptop": "笔记本电脑",
    "notebook": "笔记本",
    "desktop": "台式电脑",
    "monitor": "显示器",
    "screen": "屏幕",
    "keyboard": "键盘",
    "mouse": "鼠标",
    "tablet": "平板电脑",
    "phone": "手机",
    "smartphone": "智能手机",
    "iphone": "iPhone",
    "android": "安卓",
    "device": "设备",
    "gadget": "小工具",
    "camera": "相机",
    "video camera": "摄像机",
    "microphone": "麦克风",
    "speaker": "扬声器",
    "headphone": "耳机",
    "headset": "耳机",
    "television": "电视",
    "tv": "电视",
    "screen": "屏幕",
    "display": "显示屏",
    "projector": "投影仪",
    "console": "游戏机",
    "gamepad": "游戏手柄",
    
    # 艺术相关
    "art": "艺术",
    "artwork": "艺术品",
    "painting": "绘画",
    "drawing": "画画",
    "sketch": "素描",
    "illustration": "插图",
    "poster": "海报",
    "print": "版画",
    "sculpture": "雕塑",
    "statue": "雕像",
    "bronze": "青铜",
    "marble": "大理石",
    "ceramic": "陶瓷",
    "pottery": "陶器",
    "craft": "手工艺",
    "handmade": "手工制作",
    "vintage": "古董",
    "antique": "古董",
    "replica": "复制品",
    "display": "展示",
    "exhibit": "展览",
    "gallery": "画廊",
    "museum": "博物馆",
    "canvas": "画布",
    "easel": "画架",
    "brush": "画笔",
    "palette": "调色板",
    
    # 其他常见标签
    "person": "人物",
    "hair": "头发",
    "head": "头部",
    "hand": "手",
    "hands": "手",
    "foot": "脚",
    "feet": "脚",
    "leg": "腿",
    "legs": "腿",
    "arm": "手臂",
    "arms": "手臂",
    "back": "背部",
    "front": "正面",
    "side": "侧面",
    "body": "身体",
    "reflection": "反射",
    "mirror": "镜子",
    "shadow": "影子",
    "sunset": "日落",
    "backlit": "逆光",
    "silhouette": "轮廓",
    "profile": "侧面",
    "headshot": "头像",
    "full body": "全身",
    "closeup": "特写",
    "wide angle": "广角",
    "overhead": "俯视",
    "ground level": "地平线",
    "underwater": "水下",
    "aerial": "空中",
    "texture": "纹理",
    "pattern": "图案",
    "symmetry": "对称",
    "composition": "构图",
    "leading lines": "引导线",
    "frame": "框架",
    "border": "边框",
    "background": "背景",
    "foreground": "前景",
    "depth": "深度",
    "layers": "图层",
    "bokeh": "虚化",
    "blur": "模糊",
    "motion blur": "运动模糊",
    "sharp": "锐利",
    "focus": "焦点",
    "exposure": "曝光",
    "white balance": "白平衡",
    "color": "颜色",
    "grayscale": "灰度",
    "black and white": "黑白",
    "monochrome": "单色",
    "vibrant": "鲜艳",
    "muted": "柔和",
    "warm": "温暖",
    "cool": "清爽",
    "contrast": "对比",
    "brightness": "亮度",
    "shadow": "影子",
    "highlight": "高光",
    "reflection": "倒影",
    "refraction": "折射",
    "transparent": "透明",
    "opaque": "不透明",
    "clear": "清晰",
    "hazy": "朦胧",
    "atmospheric": "大气",
    "moody": "氛围",
    "cinematic": "电影感",
    "artistic": "艺术",
    "abstract": "抽象",
    "surreal": "超现实",
    "fantasy": "幻想",
    "dreamlike": "梦幻",
    "whimsical": "异想天开",
    "playful": "顽皮",
    "serious": "严肃",
    "dramatic": "戏剧",
    "minimalist": "极简",
    "maximalist": "极繁",
    "symmetrical": "对称",
    "asymmetrical": "非对称",
    "balanced": "平衡",
    "dynamic": "动态",
    "static": "静态",
    "active": "活跃",
    "passive": "被动",
    "energetic": "充满活力",
    "calm": "平静",
    "peaceful": "宁静",
    "serene": "宁静",
    "chaotic": "混乱",
    "orderly": "有序",
    "geometric": "几何",
    "organic": "有机",
    "structured": "结构化",
    "flowing": "流畅",
}


class TranslationService:
    """Service for translating English labels to Chinese with advanced matching."""
    
    def __init__(self, translations: Optional[dict] = None):
        """Initialize translation service.
        
        Args:
            translations: Dictionary mapping English labels to Chinese.
                         If None, uses EXTENDED_LABEL_TRANSLATIONS.
        """
        self.translations = translations or EXTENDED_LABEL_TRANSLATIONS
        # Create a lowercase lookup dictionary for case-insensitive matching
        self.lowercase_lookup = {k.lower(): v for k, v in self.translations.items()}
    
    def translate(self, label: str, language: str = "en") -> str:
        """Translate a label to the specified language.
        
        Args:
            label: The English label to translate
            language: Target language ('en' for English, 'zh' for Chinese)
            
        Returns:
            Translated label if found, otherwise original label
        """
        if language != "zh" or not isinstance(label, str):
            return label
        
        # Try exact match first
        exact_match = self._exact_match(label)
        if exact_match:
            return exact_match
        
        # Try word-by-word translation for compound labels
        translated = self._translate_compound(label)
        if translated != label:
            return translated
        
        # Try fuzzy matching as last resort
        fuzzy_match = self._fuzzy_match(label)
        if fuzzy_match:
            return fuzzy_match
        
        return label
    
    def _exact_match(self, label: str) -> Optional[str]:
        """Try exact match (case-insensitive) on the label."""
        key = label.strip().lower()
        
        # Direct lookup
        if key in self.lowercase_lookup:
            return self.lowercase_lookup[key]
        
        # Try replacing underscores/hyphens with spaces
        key_spaces = key.replace("_", " ").replace("-", " ").strip()
        if key_spaces in self.lowercase_lookup:
            return self.lowercase_lookup[key_spaces]
        
        return None
    
    def _translate_compound(self, label: str) -> str:
        """Translate compound words by breaking them into parts."""
        key = label.strip().lower()
        key_spaces = key.replace("_", " ").replace("-", " ").strip()
        
        # Split into words
        parts = key_spaces.split()
        if not parts:
            return label
        
        translated_parts = []
        for part in parts:
            if not part:
                continue
            
            # Try to find exact match for each part
            if part in self.lowercase_lookup:
                translated_parts.append(self.lowercase_lookup[part])
            else:
                # Keep original if no translation found
                translated_parts.append(part)
        
        # If all parts were translated, return the joined result
        if translated_parts and any(p in self.lowercase_lookup.values() for p in translated_parts):
            return "_".join(translated_parts)
        
        return label
    
    def _fuzzy_match(self, label: str, threshold: float = 0.6) -> Optional[str]:
        """Try fuzzy matching using sequence matching.
        
        Args:
            label: The label to match
            threshold: Minimum similarity score (0.0 to 1.0)
            
        Returns:
            Translated label if match found, None otherwise
        """
        key = label.strip().lower()
        best_match = None
        best_score = threshold
        
        for trans_key in self.lowercase_lookup.keys():
            # Use difflib for similarity matching
            similarity = difflib.SequenceMatcher(None, key, trans_key).ratio()
            if similarity > best_score:
                best_score = similarity
                best_match = trans_key
        
        return self.lowercase_lookup.get(best_match) if best_match else None
    
    def translate_labels(self, labels: list[str], language: str = "en") -> list[str]:
        """Translate a list of labels.
        
        Args:
            labels: List of English labels
            language: Target language
            
        Returns:
            List of translated labels
        """
        return [self.translate(label, language) for label in labels]


# Global translation service instance
_translation_service = None


def get_translation_service() -> TranslationService:
    """Get or create the global translation service instance."""
    global _translation_service
    if _translation_service is None:
        _translation_service = TranslationService(EXTENDED_LABEL_TRANSLATIONS)
    return _translation_service
