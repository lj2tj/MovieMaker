# -*- coding: gbk -*-
# ai_helper.py

import yaml
import json
from typing import Dict, Any, List, Tuple, Union
import os

# Langchain imports for web scraping
try:
    from langchain_community.document_loaders import WebBaseLoader
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("Langchain not available, web scraping features will be limited")


class MovieMakerAIHelper:
    """
    AI Helper 类，支持 Ollama 和 OpenAI 两种平台
    根据 config.yaml 中的 ai_helper 配置自动选择
    """
    
    def __init__(self):
        """
        初始化 AI Helper
        """
        # Load config from config.yaml
        self.config = self._load_config()
        
        # Determine which AI platform to use
        self.ai_platform = self.config.get('ai_helper', 'ollama').lower()
        print(f"使用 AI 平台: {self.ai_platform}")
        
        # Initialize client based on platform
        self.client = None
        self.chat_model = None
        self.image_model = None
        
        if self.ai_platform == 'openai':
            openai_api_key = os.getenv('OPENAI_API_KEY', self.config.get('openai_api_key', ''))
            openai_base_url = os.getenv('OPENAI_API_BASE_URL', self.config.get('openai_api_base_url', ''))
            if openai_api_key:
                from openai import OpenAI
                self.client = OpenAI(api_key=openai_api_key, base_url=openai_base_url)
                self.chat_model = os.getenv('OPENAI_CHAT_MODEL', self.config.get('openai_chat_model', 'DeepSeek-V4-Flash'))
                self.image_model = os.getenv('OPENAI_IMAGE_MODEL', self.config.get('openai_image_model', 'DeepSeek-V4-Flash'))
                print(f"OpenAI client initialized: {openai_base_url}, model: {self.chat_model}, image model: {self.image_model}")
            else:
                raise ValueError("OpenAI API key is not set")
        else:
            from ollama import Client
            ollama_host = os.getenv('OLLAMA_HOST', self.config.get('ollama_host', 'http://localhost:11434'))
            self.client = Client(host=ollama_host)
            self.chat_model = os.getenv('OLLAMA_CHAT_MODEL', self.config.get('ollama_chat_model', 'llama3'))
            self.image_model = os.getenv('OLLAMA_IMAGE_MODEL', self.config.get('ollama_image_model', 'llama3'))
            print(f"Ollama client initialized: {ollama_host}, model: {self.chat_model}, image model: {self.image_model}")
        
        # Common attributes
        self.examples = self._load_examples()
        self.actions = [
            {'名称': 'BGM', '字幕': [['', '', 'bgm', 'resources/ShengYin/BGM/xxx.mp3']], '渲染顺序': 0},
            {'名称': '镜头', '角色': '角色名称', '持续时间': '2秒', '焦点': [0.5, 0.5], '变化': 0.3, '字幕': [['', '', '台词', 'resources/ShengYin/xxx.mp3']], '渲染顺序': 1},
            {'名称': '消失', '角色': '角色名称', '渲染顺序': 2},
            {'名称': '显示', '角色': '角色名称', '渲染顺序': 3},
            {'名称': '打斗', '角色': '角色1 角色2', '幅度': "小", '字幕': [['', '', '台词', 'resources/ShengYin/xxx.mp3']], '渲染顺序': 4},
            {'名称': 'gif', '素材': 'resources/SuCai/GIF/xxx.gif', '发音人引擎': 'chat', '字幕': [['', '', '台词', 'resources/ShengYin/xxx.mp3']], '位置': [0.5, 0.5], '图层': 100, '角度': 0, '大小': [300, 300], '渲染顺序': 5},
            {'名称': '静止', '持续时间': '2秒', '字幕': [['', '', '台词', 'resources/ShengYin/xxx.mp3']], '渲染顺序': 6},
            {'名称': '转场', '背景': 'resources/beijing/xxx.jpg', '方式': '旋转缩小', '字幕': [['','', '', 'resources/水浒传/声音/回忆转场.mp3']], '渲染顺序': 7},
            {'名称': '说话', '角色': '角色名称', '焦点': [0.5, 0.5], '高亮': '是', '变化': 0.3, '字幕': [['', '', '台词', 'resources/ShengYin/xxx.mp3']], '渲染顺序': 8},
            {'名称': '转身', '角色': '角色名称', '持续时间': '1秒', '角度': 90, '字幕': [['', '', '台词', 'resources/ShengYin/xxx.mp3']], '渲染顺序': 9},
            {'名称': '更新', '角色': '角色名称', '素材': 'resources/人物/xxx.png', '角度': 0, '大小': [300, 300], '透明度': 0.5, '渲染顺序': 10},
            {'名称': '队列', '角色': '角色1 角色2', '持续时间': '1', '开始位置': [0.5, 0.5], '结束位置': [0.5, 0.5], '开始角度': 0, '结束角度': 90, 'x': 0.5, 'y': 0.5, '结束消失': '是', '结束图层': 100, '延迟': 0.5, '比例': 0.5, '方式': '自然', '字幕': [['', '', '台词', 'resources/ShengYin/xxx.mp3']], '渲染顺序': 11},
            {'名称': '行进', '角色': '角色名称', '持续时间': '1', '开始位置': [0.5, 0.5], '结束位置': [0.5, 0.5], '开始角度': 0, '结束角度': 90, 'x': 0.5, 'y': 0.5, '结束消失': '是', '结束图层': 100, '延迟': 0.5, '比例': 0.5, '方式': '自然', '字幕': [['', '', '台词', 'resources/ShengYin/xxx.mp3']], '渲染顺序': 12},
        ]
    
    def _load_config(self):
        """Load config file"""
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        return {}
    
    def _load_examples(self):
        """Load existing config examples for reference"""
        examples = []
        demo_dir = "demo"
        if os.path.exists(demo_dir):
            for file in os.listdir(demo_dir):
                if file.endswith('.yaml'):
                    with open(os.path.join(demo_dir, file), 'r', encoding='utf-8') as f:
                        examples.append(yaml.safe_load(f))
            return examples[:2]
        return []
    
    def _chat_completion(self, messages: List[Dict[str, str]], temperature: float = 0.5, **kwargs) -> str:
        """
        统一的聊天完成接口，根据平台自动选择实现
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            **kwargs: 其他参数
            
        Returns:
            生成的文本内容
        """
        if self.client is None:
            raise RuntimeError(f"{self.ai_platform} client not initialized")
        
        if self.ai_platform == 'openai':
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=messages,
                temperature=temperature,
                **kwargs
            )
            return response.choices[0].message.content
        else:
            # Ollama
            response = self.client.chat(
                model=self.chat_model,
                messages=messages,
                options={"temperature": temperature, **kwargs}
            )
            return response['message']['content']
    
    def generate_scenario_from_description(self, scenario_description: str) -> Dict[str, Any]:
        """
        Generate scenario config from description
        """
        example = self.examples[0] if self.examples else {}

        prompt = f"""
        根据以下场景描述，生成MovieMaker兼容的YAML配置片段。

        场景描述: {scenario_description}

        参考配置示例:
        {json.dumps(example, ensure_ascii=False, indent=2)}

        请严格按照MovieMaker的配置格式生成，包含场景、角色、活动和动作等元素。
        确保所有路径使用相对路径，并使用适当的坐标值（0-1之间表示百分比位置）。
        为每个角色和活动生成合适的名称和描述。
        """

        system_content = """
        你是一个MovieMaker配置文件专家。MovieMaker使用YAML格式定义视频场景，
        包括背景、角色、活动和动作。你需要生成符合MovieMaker格式的YAML配置。

        重要约束：
        - 位置坐标使用[0-1]范围内的浮点数表示相对位置
        - 大小使用像素值如[100, 150]
        - 音频文件路径应指向适当的位置
        - 每个场景包含背景、角色和活动
        - 活动包含动作和字幕
        """

        try:
            content = self._chat_completion([
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt}
            ], temperature=0.5)
            
            return yaml.safe_load(content)
        except Exception as e:
            print(f"生成场景配置失败: {e}")
            return {}

    def extract_character(self, text: str, output_dir: str = "scripttemplate/角色") -> List[str]:
        """
        Extract characters from text
        """
        prompt = f"""
        分析以下文字，提取所有角色信息。
        对于每个角色，提供：
        - 名字
        - 性别（男/女）
        - 年龄
        - 外貌特征
        - 性格特点

        文字内容：
        {text}

        请以Yaml格式返回角色列表，格式如下：
        角色:
          - 名字: 角色名
            性别: 男/女
            年龄: 年龄描述
            外貌特征: 外貌描述
            性格特点: 性格描述
        """

        try:
            content = self._chat_completion([
                {"role": "system", "content": "你是一个角色分析专家，擅长从文字中提取角色信息。"},
                {"role": "user", "content": prompt}
            ], temperature=0.5)

            import re
            yaml_match = re.search(r'```(?:yaml)?\s*\n(.*?)\n```', content, re.DOTALL)
            if yaml_match:
                yaml_content = yaml_match.group(1)
            else:
                yaml_content = content

            data = yaml.safe_load(yaml_content)
            characters = data.get('角色', [])

            os.makedirs(output_dir, exist_ok=True)
            saved_names = []

            for char in characters:
                char_name = char.get('名字', '未知角色')
                char_yaml = {'角色': [char]}
                filepath = os.path.join(output_dir, f"{char_name}.yaml")
                with open(filepath, 'w', encoding='utf-8') as f:
                    yaml.dump(char_yaml, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
                saved_names.append(char_name)
                print(f"已保存角色: {char_name}")

            return saved_names

        except Exception as e:
            print(f"角色提取失败: {e}")
            return []

    def extract_scenario(self, text: str, output_dir: str = "scripttemplate", filename: str = "场景.yaml") -> List[str]:
        """
        Analyze text and extract story outline, 
        then create multiple scenarios based on the plot.
        """
        prompt = f"""
        理解项目支持的动作类型：{self.actions}
        分析以下文字，浓缩故事情节。
        提取到的故事情节要足够简练，要包括背景、角色、台词、天气/时间、活动描述，能够覆盖文字的主要内容。
        如果有必要，可以增加特殊角色，如武器、道具、植物、载具、动物等。
        如果有必要，可以增加动作，如打斗、行进、对话，GIF等。
        字幕使用['','', '台词', '音频文件路径']格式。
        尽可能使用'说话'动作而不使用活动的'字幕'。
        各个场景之间适当使用转场动作
        并根据故事情节创建3-5个场景（scenario）。

        文字内容：
        {text}

        每个场景需要包含：
        - 场景名称
        - 场景描述
        - 主要角色（该场景中出现的角色）
        - 背景场景（室内/室外/地点描述）
        - 天气/时间
        - 活动描述（该场景中发生的主要动作和对话）

        请以Yaml格式返回场景列表，格式如下：
场景: # 每个场景共用一套角色和背景，不同的场景使用不同的角色和背景
-
  背景: resources/水浒传/背景/城墙外.png
  名字: '京城外'
  焦点: "中心"
  背景音乐: 
  比例: 1
  角色:
    -
      名字: 路人甲
      素材: resources/水浒传/人物/配角/卡拉米1.png
      位置: [0.2, 0.65]
      大小: [80, 110]
      显示: 是
      图层: 0
      角度: 45
  活动:
  -
    名字: "凄惨场景"
    描述: 展示瘟疫盛行时的凄惨场景
    背景音乐: 
    持续时间: 
    字幕: #Kangkang, Male
    - ['','', '大宋仁宗时期', 'resources/水浒传/001/京城外/声音/大宋仁宗时期.wav']
    字幕样式: list
    字幕颜色: black
    发音人: aisjiuxu
-
  背景: resources/水浒传/背景/宫殿外.png
  名字: '金殿外'
  焦点: "中心"
  背景音乐: 
  比例: 1
  角色:
  活动:
  -
    名字: "镜头移动"
    描述: 金殿外，旁白
    背景音乐: 
    持续时间: 
    字幕:  #Kangkang, Male
    字幕颜色: black
    fps: 6
    动作:
    -
      名称: 镜头
      角色: 
      持续时间: 
      焦点: [0.5, 0.5]
      变化: [1, 0.6]
      字幕: 
      - ['','', '此时', 'resources/水浒传/001/金殿外/声音/此时.wav']
      渲染顺序: 0

        只返回Yaml格式，不要包含其他文字。
        """

        try:
            content = self._chat_completion([
                {"role": "system", "content": "你是一个故事分析专家，擅长从文字中提取故事梗概和场景信息，并能够为每个场景设计合适的动作序列。"},
                {"role": "user", "content": prompt}
            ], temperature=0.5)

            # Try to parse YAML directly
            data = yaml.safe_load(content)
            
            # Handle YAML format
            if isinstance(data, dict) and '场景' in data:
                scenarios = data.get('场景', [])
                story_summary = data.get('故事梗概', '')
            elif isinstance(data, list):
                scenarios = data
                story_summary = ''
            else:
                # Fallback: try to find YAML code block
                import re
                yaml_pattern = r'```(?:yaml)?\s*\n(.*?)\n```'
                yaml_match = re.search(yaml_pattern, content, re.DOTALL)
                if yaml_match:
                    data = yaml.safe_load(yaml_match.group(1))
                    scenarios = data.get('场景', []) if isinstance(data, dict) else data
                    story_summary = data.get('故事梗概', '') if isinstance(data, dict) else ''
                else:
                    print("无法解析响应格式")
                    return []

            os.makedirs(output_dir, exist_ok=True)

            # Load existing characters from scripttemplate/角色 folder
            characters_dir = os.path.join(output_dir, '角色')
            existing_characters = {}
            if os.path.exists(characters_dir):
                for file in os.listdir(characters_dir):
                    if file.endswith('.yaml'):
                        char_name = os.path.splitext(file)[0]
                        try:
                            with open(os.path.join(characters_dir, file), 'r', encoding='utf-8') as f:
                                char_data = yaml.safe_load(f)
                                if char_data and '角色' in char_data and len(char_data['角色']) > 0:
                                    existing_characters[char_name] = char_data['角色'][0]
                                    print(f"已加载角色: {char_name}")
                        except Exception as e:
                            print(f"加载角色文件失败 {file}: {e}")

            # Create scenario YAML structure
            scenario_yaml = {
                '场景': []
            }

            scenario_names = []
            for idx, scenario in enumerate(scenarios):
                scenario_name = scenario.get('名字', f'场景{idx + 1}')
                scenario_names.append(scenario_name)

                scenario_entry = {
                    '名字': scenario_name,
                    '描述': scenario.get('描述', ''),
                    '焦点': scenario.get('焦点', '中心'),
                    '比例': scenario.get('比例', 1),
                    '天色': scenario.get('天色', ''),
                    '背景': scenario.get('背景', f'resources/背景/{scenario_name}.png'),
                    '背景音乐': scenario.get('背景音乐', None),
                    '角色': [],
                    '活动': []
                }

                # Copy characters directly from YAML response
                scenario_entry['角色'] = scenario.get('角色', [])

                # Copy activities directly from YAML response  
                scenario_entry['活动'] = scenario.get('活动', [])

                scenario_yaml['场景'].append(scenario_entry)

            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump(scenario_yaml, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            
            print(f"已保存场景文件: {filepath}")
            print(f"故事梗概: {story_summary}")
            print(f"共创建 {len(scenarios)} 个场景")

            return scenario_names

        except Exception as e:
            print(f"解析失败: {e}")
            return []

    def create_image(self, description: str, width: int, height: int, output_path: str) -> bool:
        """
        Generate an image based on description
        """
        if self.client is None:
            print(f"{self.ai_platform} client not initialized")
            return False
        
        # Check if image already exists
        if os.path.exists(output_path):
            print(f"图片已存在: {output_path}")
            return False

        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        try:
            print(f"正在生成图片: {output_path}")
            print(f"描述: {description}")
            
            if self.ai_platform == 'openai':
                # OpenAI image generation
                # Note: This is a placeholder - actual implementation depends on OpenAI's API
                print("OpenAI image generation not yet implemented")
                return False
            else:
                # Ollama image generation
                response = self.client.chat(
                    model=self.image_model,
                    messages=[{
                        'role': 'user',
                        'content': f"what is the image",
                        'images': ['resources\u6c34\u6d52\u4f20\u80cc\u666f\u57ce\u5899\u5916.png']
                    }]
                )
                print(response)
                return False

        except Exception as e:
            print(f"生成图片失败: {e}")
            return False

    def extract_text_from_webpage(self, url: str, chunk_size: int = 2000, chunk_overlap: int = 200) -> Tuple[str, str]:
        """
        Extract text from a webpage using Langchain.
        
        Args:
            url: URL of the webpage to extract text from
            chunk_size: Size of each text chunk
            chunk_overlap: Overlap between consecutive chunks
            
        Returns:
            Tuple of (title, content)
        """
        if not LANGCHAIN_AVAILABLE:
            print("Langchain is not available, please install it first: pip install langchain")
            return "", ""

        try:
            print(f"正在从网页提取文字: {url}")
            
            # Use Langchain's WebBaseLoader to load the webpage
            loader = WebBaseLoader(url)
            documents = loader.load()
            
            if not documents:
                print("未能从网页加载内容")
                return "", ""
            
            return documents[0].metadata["title"], documents[0].page_content
            
        except Exception as e:
            print(f"从网页提取文字失败: {e}")
            return "", ""


# Factory function for backward compatibility
def create_ai_helper() -> MovieMakerAIHelper:
    """
    工厂方法：创建 AI Helper 实例
    
    Returns:
        MovieMakerAIHelper 实例
    """
    return MovieMakerAIHelper()


# Example usage
if __name__ == "__main__":
    helper = MovieMakerAIHelper()
    url = "https://www.gushicimingju.com/novel/shuihuzhuan/676.html"
    title, story_text = helper.extract_text_from_webpage(url, header_template={})
    
    if not title or not story_text:
        print("未能从网页提取标题或内容")
        exit(1)

    if "上一章：" in story_text:
        story_text = story_text.split("上一章：")[0]
    if "下一章：" in story_text:
        story_text = story_text.split("下一章：")[0]

    print(f"标题: {title}")
    print(f"内容: {story_text}")

    scenarios = helper.extract_scenario(story_text, 
                                        output_dir="scripttemplate/场景",
                                        filename="场景.yaml")
    
    print(f"提取到的场景: {scenarios}")

    # desc = 'this is a character image with width 160 and height 240. the character instructor of the imperial troops during the Song Dynasty of China, with a stern and weathered face, eyes interweaving restrained and eruptive anger, brows tightly furrowed, mouth firmly pressed. He wears silver armor, over which is a blue battle robe soaked by wind and snow, shoulder carrying a Zhuanga snake spear, with an ancient wine gourd hanging from the spear tip.',
    # helper.create_image(
    #     description=desc, 
    #     width=160, 
    #     height=240, 
    #     output_path="scripttemplate/instructor.jpg")
