# 项目库开发流程
Git文件版本控制, 以commit为单元记录文件更改, 通过操作单个commit实现文件的更改, 回溯等操作.   
通过不同branch并存, 实现库的多个版本控制.   
同时通过github等远程库与本地库的branch互联, 实现多人协作开发. 
## 本地开发流程

1. 当需要在本地设备进行开发时, 首先`git clone`待开发的代码库;
2. 然后`git checkout -b local`创建本地local分支；
4. 在本地local分支上进行开发, 将更改commit，可发布远程origin/local分支；
5. 往后开发保持local分支和远程origin/local分支的互联；

## 本地库与远程库
当在本地库local开发时，需要注意以下:
- 本地local分支的所有commit，及时push到远程origin/local分支；
- 避免直接从他人远程origin/local分支直接fetch, 并应该在每次开发前fetch远程origin/main, 确保本地开发时的基为最新；

## 单人库维护与多人库维护
单人库维护与多人库维护的区别主要在于：**如何将本地开发提交到远程正式库**
- **单人库维护**:
  1. `local1`分支完成开发和测试后，将`local1`分支合并到`main`分支；
  2. 将`main`分支推送到远程`origin/main`分支；
- 注意：
  - 此时远程`origin/local`分支一般没有，或只有一个。可以从远程`origin/local`分支拉取作为`local`分支的基，以继续开发

- **多人库维护**：
  1. `local1`分支完成开发后，将local分支推送到远程`origin/local1`分支；
  2. 测试远程`origin/local1`分支，若测试通过，创建`Pull Request`；
  3. 经正式库管理者同意`Pull Request`，此时即将远程`origin/local1`分支合并到远程`origin/main`分支；
- 注意：
  - 此时远程`origin/local`分支一般存在多个。一般应避免拉取他人远程`origin/local`分支
