使用步骤

1.安装labelme，使用 pip install labelme 命令即可。

2.在labelme环境下输入 labelme命令，打开labelme软件，对图片进行标记，具体方法就是用多边形将所有水体部分圈起来命名为water，并保存文件(json格式)。

3.将main.m文件内fname和imagename改为对应的json文件名和image文件名，之后使用matlab运行main.m文件，稍等片刻，即可看到训练过程，训练结束后可以看到ac率和预测后图像和原始图像的对比。

labelme的GitHub地址：https://github.com/wkentaro/labelme
