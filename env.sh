# 升级pip
python -m pip install --upgrade pip

# 下载模型
# 定义本地目录和文件名  
local_dir="/root/data/model/sentence-transformer"  
model_name="paraphrase-multilingual-MiniLM-L12-v2"  
  
# 检查本地文件是否存在  
if [ ! -f "$local_dir/$model_name" ]; then  
    # 如果文件不存在，则执行下载命令  
    echo "开始下载Embedding模型"  
    huggingface-cli download --resume-download sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --local-dir $local_dir  
else
    echo "Embedding模型已存在，跳过下载模型操作。"  
fi

#下载nltk数据集
# 切换到root目录  
cd /root  
# 检查nltk_data目录是否存在  
if [ -d "nltk_data" ]; then  
    echo "nltk_data目录已存在，跳过克隆操作。"  
else  
    # 克隆指定的git仓库到本地，使用指定的分支  
    git clone https://gitee.com/yzy0612/nltk_data.git --branch gh-pages  
    # 切换到克隆下来的仓库目录  
    cd nltk_data  
    # 将packages目录下的所有文件移动到当前目录  
    mv packages/* ./  
    # 切换到tokenizers目录  
    cd tokenizers  
    # 解压punkt.zip文件  
    unzip punkt.zip  
    # 切换到taggers目录  
    cd ../taggers  
    # 解压averaged_perceptron_tagger.zip文件  
    unzip averaged_perceptron_tagger.zip
fi  
