## step
1. python 3.10；太新的东西有的时候不支持（所以演示用了 3.10），但是学习的话还是建议大家用 3.12(多踩踩坑，这样就会多点到 langchain 源码中)

2. pip install -U openai langchain langchain-core python-dotenv loguru tyro
然后遇到什么 库依赖报错，安装对应的库
用法：`pip install xxx` （xxx 表示对应的库，可以去 requirements.txt 中查看）

3. pip install "langserve[all]"


> 注释：
>
> pip install -r requirements.txt  （不推荐，这里只是给大家参考我装了什么库）
>
> 这里的 httpx 版本可能需要特殊注意一下；因为 httpx 自带 bug, 所以安装这个版本 

