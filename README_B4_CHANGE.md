# 项目库开发流程
## 本地开发流程
1. 当需要在本地设备进行开发时，首先创建本地开发`local1`分支；
2. 从远程`origin/main`分支拉取正式库，以更新`local1`分支；
3. 进行更改，更改后发布远程`origin/local1`分支；
4. 往后开发保持`local1`分支和远程`origin/local1`分支的互联；

## 本地库与远程库
当在本地库local1开发时，需要注意以下：
- 确保`local1`分支的基为最新的远程origin/main分支；
- `local1`分支的所有更改在开发完成后，及时推送到远程`origin/local1`分支；
- 避免直接从他人远程`origin/local2`分支等直接拉取未经测试的更改；

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