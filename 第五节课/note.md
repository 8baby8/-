# LMDeploy 的量化和部署

## 1、环境配置
两种方法，克隆带有torch 2.0.1的环境。
```bash
# 方法1
conda create -n lmdeploy --clone /share/conda_envs/internlm-base
# 方法2 
/share/install_conda_env_internlm_base.sh lmdeploy
```
 
环境克隆成功之后激活环境。

```bash
conda activate lmdeploy
```
接下来安装 lmdeploy
```bash
#分别执行以下两行命令
pip install /root/share/wheels/flash_attn-2.4.2+cu118torch2.0cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
pip install 'lmdeploy[all]==v0.1.0'
```
注意：测试lmdeploy是否安装成功，如果找不到lmdeploy这个命令在终端输入
```bash
export PATH=$PATH:'/root/.local/bin'
```
至此，基本环境已完成.
## 2.服务部署
![img](assets/lmdeploy.drawio.png)

- 模型推理/服务。主要提供模型本身的推理，一般来说可以和具体业务解耦，专注模型推理本身性能的优化。可以以模块、API等多种方式提供。
- Client。可以理解为前端，与用户交互的地方。
- API Server。一般作为前端的后端，提供与产品和服务相关的数据和功能支持。

### 2.1 模型转换
使用 TurboMind 推理模型需要先将模型转化为 TurboMind 的格式，目前支持在线转换和离线转换两种形式。在线转换可以直接加载 Huggingface 模型，离线转换需需要先保存模型再加载。

TurboMind 是一款关于 LLM 推理的高效推理引擎，基于英伟达的 FasterTransformer 研发而成。它的主要功能包括：LLaMa 结构模型的支持，persistent batch 推理模式和可扩展的 KV 缓存管理器。
#### 在线转换
lmdeploy 支持直接读取 Huggingface 模型权重，目前共支持三种类型：

在 huggingface.co 上面通过 lmdeploy 量化的模型，如 llama2-70b-4bit, internlm-chat-20b-4bit
huggingface.co 上面其他 LM 模型，如 Qwen/Qwen-7B-Chat

such as：
```bash
# 需要能访问 Huggingface 的网络环境注意注意，这是前提
# 加载使用 lmdeploy 量化的版本
lmdeploy chat turbomind internlm/internlm-chat-20b-4bit --model-name internlm-chat-20b
# 加载其他 LLM 模型。
lmdeploy chat turbomind Qwen/Qwen-7B-Chat --model-name qwen-7b

# 启动本地模型
lmdeploy chat turbomind /share/temp/model_repos/internlm-chat-7b/  --model-name internlm-chat-7b
```

#### 离线转换（注意需要将模型转换为 lmdeploy TurboMind 模式，我简称为涡轮增压模式哈哈哈）

```bash
# 两个参数，一个名称，一个模型路径
lmdeploy convert internlm-chat-7b  /root/share/temp/model_repos/internlm-chat-7b/
```
执行完成后将会在当前目录生成一个 workspace 的文件夹。这里面包含的就是 TurboMind 和 Triton “模型推理”需要到的文件。
此时workspace的结构如下：
![img](assets/workspace目录结构.png)

weights 和 tokenizer 目录分别放的是分层拆分后的参数和 Tokenizer。
可以使用命令查看
```bash
ll workspace/triton_models/weights/
```
![img](assets/模型分层拆分.png)
每一份参数第一个 0 表示“层”的索引，后面的那个0表示 Tensor 并行的索引，因为我们只有一张卡，所以被拆分成 1 份。如果有两张卡可以用来推理，则会生成0和1两份，也就是说，会把同一个参数拆成两份。比如 layers.0.attention.w_qkv.0.weight 会变成 layers.0.attention.w_qkv.0.weight 和 layers.0.attention.w_qkv.1.weight。执行 lmdeploy convert 命令时，可以通过 --tp 指定（tp 表示 tensor parallel），该参数默认值为1（也就是一张卡）。


### 2.2 涡轮增压推理
#### 命令行启动客户端
可以直接在终端进行对话
```bash
lmdeploy chat turbomind ./workspace
```

### 2.3 TurboMind推理+API服务
模型推理/服务“目前提供了 Turbomind 和 TritonServer 两种服务化方式。此时，Server 是 TurboMind 或 TritonServer，API Server 可以提供对外的 API 服务。我们推荐使用 TurboMind

```bash
# ChatApiClient+ApiServer（注意是http协议，需要加http）
lmdeploy serve api_client http://localhost:23333
```
启动的FastAPi服务
### 2.4 网页 Demo 演示
这个是端口映射，本地端口，和远程服务器端口
ssh -CNg -L 6006:127.0.0.1:6006 root@ssh.intern-ai.org.cn -p <你的 ssh 端口号>
#### TurboMind 服务作为后端

启动的gradio前端
```bash
# Gradio+ApiServer。必须先开启 Server，此时 Gradio 为 Client
lmdeploy serve gradio http://0.0.0.0:23333 \
	--server_name 0.0.0.0 \
	--server_port 6006 \
	--restful_api True
```
效果如下
![img](assets/17.png)

#### TurboMind 推理作为后端
当然，Gradio 也可以直接和 TurboMind 连接，如下所示。
```bash
# Gradio+Turbomind(local)
lmdeploy serve gradio ./workspace
```
可以直接启动 Gradio，此时没有 API Server，TurboMind 直接与 Gradio 通信。

### 2.5 TurboMind 推理 + Python 代码集成
lmdeploy 还支持 Python 直接与 TurboMind 进行交互，如下所示。
```python
from lmdeploy import turbomind as tm
# load model
model_path = "/root/share/temp/model_repos/internlm-chat-7b/"
tm_model = tm.TurboMind.from_pretrained(model_path, model_name='internlm-chat-20b')
generator = tm_model.create_instance()

# process query
query = "你好啊! 我是朝科"
prompt = tm_model.model.get_prompt(query)
input_ids = tm_model.tokenizer.encode(prompt)

# inference
for outputs in generator.stream_infer(
        session_id=0,
        input_ids=[input_ids]):
    res, tokens = outputs[0]

response = tm_model.tokenizer.decode(res.tolist())
print(response)
```

首先加载模型，然后构造输入，最后执行推理。

加载模型可以显式指定模型路径，也可以直接指定 Huggingface 的 repo_id，还可以使用上面生成过的 workspace。这里的 tm.TurboMind 其实是对 C++ TurboMind 的封装。

构造输入这里主要是把用户的 query 构造成 InternLLM 支持的输入格式，比如上面的例子中， query 是“你好啊！我是朝科”，构造好的 Prompt 如下所示。
```bash
"""
<|System|>:You are an AI assistant whose name is InternLM (书生·浦语).
- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.

<|User|>:你好啊!朝科
<|Bot|>:
"""
```

## 3 模型量化
### 3.1.1 量化步骤
KV Cache 量化是将已经生成序列的 KV 变成 Int8，使用过程一共包括三步：

第一步：计算 minmax。主要思路是通过计算给定输入样本在每一层不同位置处计算结果的统计情况。

对于 Attention 的 K 和 V：取每个 Head 各自维度在所有Token的最大、最小和绝对值最大值。对每一层来说，上面三组值都是 (num_heads, head_dim) 的矩阵。这里的统计结果将用于本小节的 KV Cache。
对于模型每层的输入：取对应维度的最大、最小、均值、绝对值最大和绝对值均值。每一层每个位置的输入都有对应的统计值，它们大多是 (hidden_dim, ) 的一维向量，当然在 FFN 层由于结构是先变宽后恢复，因此恢复的位置维度并不相同。这里的统计结果用于下个小节的模型参数量化，主要用在缩放环节（回顾PPT内容）。