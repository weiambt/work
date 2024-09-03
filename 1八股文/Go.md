# Go



### 切片

在Go语言中，切片（slice）是一个对数组的一个连续片段的引用，它提供了对底层数组的抽象，并允许程序员像操作数组一样操作这个片段，但具有更高的灵活性和效率。切片本身并不是数据集合，而是对底层数组的引用，并包含指向该数组的指针、长度和容量。

以下是关于Go切片的一些重要点和特性：

1. **定义**：
   切片的类型表示为`[]T`，其中`T`是切片中元素的类型。例如，`[]int`表示一个整数切片。

2. **创建**：
   切片可以通过多种方式创建，最常见的是使用`make`函数或者从一个现有数组或切片中切分出来。

   使用`make`函数：

   ```go
   go复制代码
   
   s := make([]int, 5) // 创建一个长度为5的整数切片，容量也为5
   ```

   从数组切分：

   ```go
   a := [5]int{1, 2, 3, 4, 5}  
   s := a[1:4] // 创建一个从索引1（包含）到索引4（不包含）的切片
   ```

3. **长度和容量**：
   切片的长度是切片中元素的数量，而容量是从切片的第一个元素到底层数组中最后一个元素的数量。可以使用`len(s)`获取长度，使用`cap(s)`获取容量。

- 切片与数组的区别：是否指定数组长度

  `[]int` 表示一个整数切片（slice），而不是整型数组（array）。

  整数数组的类型会指定其长度，例如 `[5]int` 表示一个包含5个整数的数组

### goroutine

在Go语言中，goroutine是一种**轻量级的执行单元**，**用户级别的调度**，相比线程级别操作系统的调度（需要频繁切换上下文），性能更好，可以将其理解为一个函数的并发执行。与传统的线程相比，goroutine更加轻量级，创建和销毁的开销非常小。

Go语言的运行时系统（runtime）负责调度和管理goroutine的执行。

- 创建与使用
  -  **创建**：使用`go`关键字后跟函数调用即可创建一个新的Goroutine。例如，`go myFunction()`将启动一个新的Goroutine来执行`myFunction`函数。
  -  **执行**：当Goroutine被创建后，它将与创建它的Goroutine（通常是main Goroutine）并行执行。Goroutine的执行顺序不是确定的，取决于Go的运行时调度器。
  -  **通信**：Goroutine之间可以使用通道（channel）进行通信，以实现同步和协作。

- 例子

```
package main  
  
import (  
    "fmt"  
    "time"  
)  
  
func counter(id int) {  
    for i := 0; i < 5; i++ {  
        fmt.Println("Goroutine", id, "count:", i)  
        time.Sleep(time.Second) // 模拟耗时操作  
    }  
}  
  
func main() {  
    // 启动两个Goroutine  
    go counter(1)  
    go counter(2)  
  
    // 等待一段时间，让Goroutine有足够的时间执行  
    time.Sleep(4 * time.Second)  
    fmt.Println("Main function terminated.")  
}
```

