

# Linux命令

### grep -A -B -C 使用介绍

grep -A 显示匹配指定内容及之后的n行

grep -B  显示匹配指定内容及之前的n行

grep -C  显示匹配指定内容及其前后各n行

![img](https://img-blog.csdnimg.cn/7e9f85cf26194151931d81ea5d6625cb.png)

例子

$  grep -A  5  name  test.txt

搜索匹配test.txt文件中与”name”[字符串匹配](https://so.csdn.net/so/search?q=字符串匹配&spm=1001.2101.3001.7020)的行，并显示其后的5行

### 查询IP地址

- ifconfig
- hostname -I

### 查看CPU利用率和内存占用率

top

### 查看文件指定行

- 方法1：head+tail

  head -5 a.txt| tail -1

  我们查看文件的命令很多，常用的有 cat、 nl、head、tail、more和less。但只查看一行，一个命令不能完成，大多数命令是查看很多行（cat查看全部行；head查看前几行；tail查看后几行）。我们可以由两个常用命令组合，查看第5行：先用head  -5  file命令查看文件前5行，再通过管道|执行tail  -1命令，查看倒数一行，达到仅查看第5行之目的。
  https://blog.csdn.net/qq_36142959/article/details/132323985

- 方法2：vim

  vim 中，输入“:100G”定位到第100行

### 查看进程ID

ps -ef

### linux怎么查看占用端口号

在Linux中，可以使用`netstat`或`ss`命令来查看占用端口号的情况。

```
netstat -tulpn
```

- `-t` 表示显示TCP端口。
- `-u` 表示显示UDP端口。
- `-l` 表示显示监听状态的端口。
- `-n` 表示显示数字形式的端口和地址，不做域名解析。
- -p 显示正在使用套接字的进程。



### 根据进程id查询端口

先通过ps -ef知道了进程id，再使用netstat查询端口，最后通过管道搜索ID

```
netstat -tulpn | grep 进程ID
```



### linux将文件中包含a的行输出到b文件

```
grep 'a\|b' file.txt > bfile.txt
```





### 项目中提到要读取服务器的cpu状态，中断信息，那从哪里读取呢？，为什么要从这个目录读？

（proc子目录）

解答思路：/proc目录是一个虚拟文件系统，它提供了一个简单的接口来访问内核信息。通过读取/proc目录下的文件，可以获取系统运行时的各种信息，包括CPU状态、中断信息、内存使用情况等。对于获取服务器的CPU状态和中断信息，可以读取/proc/cpuinfo文件来获取CPU相关信息，读取/proc/interrupts文件来获取中断信息。 问题考点的深度知识讲解：在Linux系统中，/proc目录是一个伪文件系统，用于访问内核运行时的信息。其中的文件和目录并不占用磁盘空间，而是由内核动态生成的。对于需要获取系统信息的应用程序或者系统管理员来说，可以通过读取/proc目录下的文件来获取各种信息，如CPU信息、内存信息、进程信息等。/proc/cpuinfo文件包含了关于CPU的详细信息，如厂商、型号、频率等；/proc/interrupts文件包含了系统上的中断信息，包括每个中断的编号、中断处理次数等。通过读取这些文件，可以实时监控服务器的运行状态，优化系统性能，排查问题等。