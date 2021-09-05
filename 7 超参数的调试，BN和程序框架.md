## 7 超参数调试，Batch Norm和程序框架

### 7.1 调试处理

#### 7.1.1 超参数的重要性比较

**深度学习中，用到的超参数有**

$α $：学习率

$β$：动量梯度下降因子

$β1,β2,ε$：Adam算法参数

layers：神经网络层数

hidden units：各隐藏层神经元个数

learning rate decay：学习因子下降参数

mini-batch size：批量训练样本包含的样本个数



学习率$α$为最重要的超参数；

动量梯度下降因子$β$、各隐藏层神经元个数hidden units、批量训练样本包含的样本个数mini-batch size，三者重要性仅次于学习因子$α$；

接下来为神经网络层数layers和学习因子下降参数learning rate decay；最后，由于Adam算法参数$β1,β2,ε$三个参数一般设置为0.9，0.999和$10^{-8}$，不需要反复调试，所以重要性较弱。

但超参数的重要性排名不是一成不变的，可能会根据情况的不同而发生一定的变化



#### 7.1.2 超参数的调试过程

在传统的机器学习中，我们对每个参数等距离选取任意个数的点，然后，分别使用不同点对应的参数组合进行训练，最后根据验证集上的表现，来选定最佳的参数。例如有两个待调试的参数，分别在每个参数上选取5个点，这样构成了5x5=25的参数组合，如图：

![image.png](https://upload-images.jianshu.io/upload_images/24439865-362032427039d456.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

此做法在参数较少的时候效果较好。但在深度学习的神经网络中，比较好的做法是使用随机选择。即：对于上面这个例子，我们随机选择25个点，作为待调试的超参数，如图：

![image.png](https://upload-images.jianshu.io/upload_images/24439865-9db5229b93b156b2.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


随机选择参数的目的是为了尽可能地得到更多种参数组合。如上面的例子，如果使用均匀采样的话，每个参数只有5种情况；而使用随机采样的话，每个参数有25种可能的情况，因此更有可能得到最佳的参数组合。

这种做法的另一个好处是能够对重要性不同的参数之间的选择效果更好。假设hyperparameter1为$α$，hyperparameter2为$ε$，二者的重要性是不一样的。如果使用第一种均匀采样的方法，$ε$的影响很小，相当于只选择了5个$α$值。而如果使用第二种随机采样的方法，$ε$和$α$都有可能选择25种不同值。这大大增加了$α$调试的个数，选择到最优值的可能更大了。在完全不知道哪个参数更加重要的情况下，随机采样的方式可以有效解决这一问题，但是均匀采样不能够解决该问题。

在随机采样后，可以得到某些区域模型的表现较好。为了得到更精确的最佳参数，我们应该继续对选定的区域进行由粗到细的采样（coarse to fine sampling scheme）。即放大表现较好的区域，再对此区域做更密集的随机采样。例如，对下图中右下角的方形区域再做25点的随机采样，以获得最佳参数。
![image.png](https://upload-images.jianshu.io/upload_images/24439865-8ad84d5f3a188652.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### 7.2 为超参数选择合适的范围

#### 7.2.1 不同超参数合适的标尺

例如：神经网络层数layers和各隐藏层神经元个数hidden units，因为两者都为正整数，所以可以直接进行随机均匀采样（即每次变化的幅度是一样的）；而对于学习率$α$和动量梯度下降因子$β$，则需要使用非均匀随机采样。

#### 7.2.2 对数轴的取值过程

通过将linear scale转换为log scale来实现均匀尺度转化为非均匀尺度，然后再在log scale下进行均匀采样。如，[0.0001, 0.001]，[0.001, 0.01]，[0.01, 0.1]，[0.1, 1]各个区间内随机采样的超参数个数基本一致，也就扩大了之前[0.0001, 0.1]区间内采样值个数。假设线性区间为[a, b]，令$m=log(a)，n=log(b)$，则对应的log区间为[m,n]。对log区间的[m,n]进行随机均匀采样，得到的采样值$r$，再反推到线性区间，即10的r次方。则10的r次方为最终采样的超参数。

#### 7.2.3 为什么$β$不用linear scale进行取值

假设β从0.9000变化为0.9005，则$1/1−β$基本没有变化。但如果$β$从0.9990变为0.9995，那么$1/1−β$前后差别1000。即：$β$越接近1，指数加权平均的个数越多，变化越大。因此在$β$接近1的区间，应采集得更密集一些。

### 7.3 超参数训练的实践： Pandas VS Caviar

#### 7.3.1 熊猫方式Panda approach

只能够对一个模型进行训练，调试不同的超参数，使得这个模型有最佳的表现。

#### 7.3.2 鱼子酱方式Caviar approach

可以对多个模型同时进行训练，每个模型上调试不同的超参数，根据模型的表现，选择最优的模型

一般对于非常复杂或者数据量很大的模型，更多使用Panda approach。



### 7.4 归一化网络的激活函数

####  7.4.1 logistic 回归中归一化输入特征

![](https://upload-images.jianshu.io/upload_images/24408091-d9d23cf1a78476d0.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
$$
\begin{array}{l}
\mu=\frac{1}{m} \sum_{i} x^{(i)} \\
X=X-\mu \\
\sigma^{2}=\frac{1}{m} \sum_{i} x^{(i) 2} \\
X=X / \sigma^{2}
\end{array}
$$
计算平均值，从训练集中减去平均值；计算方差，根据方差归一化数据集。变化学习问题的轮廓，易于算法优化。即归一化$x_{1}$,$x_{2}$,$x_{3}$有助于更有效的训练$w$和$b$。

<img src="https://upload-images.jianshu.io/upload_images/24408091-7c353dc031e51274.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240" style="zoom: 50%;" />



#### 7.4.2 更深层的模型的归一化

![](https://upload-images.jianshu.io/upload_images/24408091-9eea49a1674e01da.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

**分析**：此例中，与上例同理，归一化$a^{[2]}$的平均值和方差以便使$w^{[3]}$,$b^{[3]}$的训练更有效率，严格来说我们真正归一化的不是$a^{[2]}$而是$z^{[2]}$。关于归一化$a^{[2]}$还是$z^{[2]}$，深度学习文献中有一些争论，实践中，常做的是**归一化$z^{[2]}$**,吴恩达推荐其为默认选择。

**实施BN**:

在神经网络中，给定一些中间值，假设你有一些隐藏单元值，从$z^{(1)}$到$z^{(m)}$，这些来源于隐藏层，写为$z^{[l](i)}$会更准确，我们省略$𝑙$及方括号，以便简化这一行的符号。

已知这些值，计算平均值，所有这些都是针对𝑙层，但省略$𝑙$及方括号，接着，取每个$z^{(i)}$值，使其归一化，如下，减去均值再除以标准偏差，为了使数值稳定，通常分母加上$𝜀$，以防$𝜎 = 0$的情况。


$$
\begin{array}{l}
\mu=\frac{1}{m} \sum_{i} z^{(i)} \\
\sigma^{2}=\frac{1}{m} \sum_{i}\left(z^{(i)}-\mu\right)^{2} \\
z_{\text {norm}}^{(i)}=\frac{z^{(i)}-\mu}{\sqrt{\sigma^{2}+\varepsilon}}
\end{array}
$$
这时，已把这些$z$值归一化，化为平均值0和标准单位方差，所以每个分量都含有平均值0和方差1，但我们不想让隐藏单元总是含有平均值0和方差1，隐藏单元有了不同的分布才会有意义。于是做如下计算：
$$
\tilde{z}^{(i)}=\gamma z_{\text {norm}}^{(i)}+\beta
$$
这里，$\gamma$和$\beta$是模型的**学习参数**，使用梯度下降或一些其他类似梯度下降的算法，如Momentum,或Nesterov,Adam更新$\gamma$和$\beta$，正如神经网络的权重。

$\gamma$和$\beta$的作用是可以随意设置$\tilde{z}^{(i)}$的平均值。当$\gamma=\sqrt{\sigma^{2}+\varepsilon}$, $\beta=\mu$时，$\tilde{z}^{(i)}=z^{(i)}$。此时，该归一化过程，即这四个等式，只是计算恒等函数。但通过对$\gamma$和$\beta$合理设定，可以构造含其他平均值和方差的隐藏单元值。在网络中，用$\tilde{z}^{(i)}$取代$z^{(i)}$,方便神经网络中的后续计算。

**归纳**：$Batch Norm$归一化的不只是输入层，同样应用于深度隐藏层。但训练输入和隐藏单元值的一个区别是，我们不希望隐藏单元值总是均值0和方差1。

<img src="https://upload-images.jianshu.io/upload_images/24408091-f7399e00e30184c1.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240" style="zoom: 40%;" />

比如我们使用sigmoid激活函数，想使它们有更大的方差或非0的均值，以便更好的利用非线性的sigmoid函数，而不是所有值都集中在线性区域中。

有了$\gamma$和$\beta$两个参数，均值和方差由其控制，使隐藏单元值的均值和方差归一化，即$z^{(i)}$有固定的均值和方差。其均值和方差可以是0和1，也可以是其他值。

### 7.5 将 Batch Norm 拟合进神经网络

#### 7.5.1 在网络中加入 Batch Norm

<img src="https://upload-images.jianshu.io/upload_images/24408091-5fb98063f3685385.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240" style="zoom: 67%;" />

![](https://upload-images.jianshu.io/upload_images/24408091-05fbabe31579a850.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)BN发生在计算 $z$ 的计算和 $a$ 之间。与其应用没有归一化的 $z$ 值，不如用归一过的 𝑧̃ 。

得到算法的新参数：
$$
\begin{array}{l}
w^{[1]}, b^{[1]}, w^{[2]}, b^{[2]}, \ldots w^{[L]}, b^{[L]} \\
\beta^{[1]}, \gamma^{[1]}, \beta^{[2]}, \gamma^{[2]}, \ldots \beta^{[L]}, \gamma^{[L]}
\end{array}
$$
接下来可以用想用的任何一种优化算法来更新它们。比如使用梯度下降法，对给定层计算$d \beta^{[l]}$接着更新参数$\beta$为$\beta^{[l]}=\beta^{[l]}-\alpha d \beta^{[l]}$。也可以使用Adam或RMSprop或Momentum等更新$\gamma$和$\beta$。

如果使用深度学习编程框架，通常不需要自己将Batch Norm步骤应用于Batch Norm层，比如说，在Tensorflow框架中，可以用函数（tf.nn.batch_normalization）来实现Batch Normalization。

#### 7.5.2 Batch Norm与mini-batch一起使用

实践中，Batch Norm通常和训练集的mini-batch一同使用。

![](https://upload-images.jianshu.io/upload_images/24408091-c9c7cc0d8a92e15a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

过程：

- 用第一个 mini-batch($X^{\{1\}}$)应用参数$w^{[1]}$,$b^{[1]}$计算$z^{[1]}$,接着BN会减去均值，除以标准差，由$\beta^{[1]}$和$\gamma^{[1]}$重新缩放，得到$\tilde{z}^{[1]}$，再应用激活函数得到$a^{[1]}$，然后用$w^{[2]}$,$b^{[2]}$计算$z^{[2]}$等等。
- 在第二、三...个mini-batch上同样这样做，继续训练。

由于计算$z$的方式：$z^{[l]}=w^{[l]} a^{[l-1]}+b^{[l]}$，而BN先将$z^{[l]}$归一化，结果为均值0和标准方差，再由$\gamma$和$\beta$缩放。这意味着，无论$b^{[l]}$的值是多少，都会被减去，因为在BN过程中，要计算$z^{[l]}$的均值再减去平均值，在此例中的mini-batch中增加任何常数，都将会被均值减去所抵消，数值不会改变。

故在使用BN时，消除参数$b^{[l]}$，或者将其设置为0，参数计算变为：
$$
z^{[l]}=w^{[l]} a^{[l-1]}\\
z_{\text {norm}}^{[l]}\\
\tilde{z}^{[l]}=\gamma^{[l]} z_{\text {norm}}^{[l]}+\beta^{[l]}
$$
最后用$\beta^{[l]}$影响转置或偏置条件。

此例中，$z^{[l]}$，$\beta^{[l]}$,$\gamma^{[l]}$维度都是$\left(n^{[l]}, 1\right)$。

#### 7.5.3 用BN来应用梯度下降法

for t = 1......num Mini Batches

​      compute forward prop on $X^{\{t\}}$

​      in each hidden layer,use BN to replace $z^{[l]}$with $\tilde{z}^{[l]}$

​      Use backprop to compute $d w^{[l]}$,$d \beta^{[l]}$,$d \gamma^{[l]}$

​      Update parameters $w^{[l]}=w^{[l]}-\alpha d w^{[l]}$, $\beta^{[l]}=\beta^{[l]}-\alpha d \beta^{[l]}$ ,$\gamma^{[l]}=\gamma^{[l]}-\alpha d \gamma^{[l]}$

​      Works with  momentum rmsprop,adam

------



- 假设使用 mini-batch梯度下降法，运行𝑡 = 1到 batch 数量的 for 循环，

- 在 mini-batch  $X^{\{t\}}$上应用正向传播，每个隐藏层都应用正向传播，使用 BN代替$z^{[l]}$为$\tilde{z}^{[l]}$。确保在这个 mini-batch 中，$𝑧$值有归一化的均值和方差，归一化均值和方差后是$\tilde{z}^{[l]}$
- 然后，反向传播计算$d w^{[l]}$，及所有 $l$ 层所有的参数，$d \beta^{[l]}$和$d \gamma^{[l]}$。
- 最后，更新这些参数$w^{[l]}=w^{[l]}-\alpha d w^{[l]}$, $\beta^{[l]}=\beta^{[l]}-\alpha d \beta^{[l]}$ ,$\gamma^{[l]}=\gamma^{[l]}-\alpha d \gamma^{[l]}$

也适用于有 Momentum、RMSprop、Adam 的梯度下降法及其它的一些优化算法来更新由 Batch 归一化添加到算法中的$\beta$和$\gamma$ 参数。

### 7.6 Batch Norm 为什么奏效

#### 7.6.1 什么是Internal Covariate Shift

Batch Normalization的原论文作者给了Internal Covariate Shift一个较规范的定义：在深层网络训练的过程中，由于网络中**参数变化**而引起内部结点**数据分布发生变化**的这一过程被称作Internal Covariate Shift。

深度神经网络之所以如此难训练，其中一个重要原因就是网络中层与层之间存在高度的关联性与耦合性。下图是一个多层的神经网络，层与层之间采用全连接的方式进行连接。

我们规定左侧为神经网络的底层，右侧为神经网络的上层。那么网络中层与层之间的关联性会导致如下的状况：随着训练的进行，网络中的参数也随着梯度下降在不停更新。一方面，当底层网络中参数发生微弱变化时，由于每一层中的线性变换与非线性激活映射，这些微弱变化随着网络层数的加深而被放大（类似蝴蝶效应）；另一方面，参数的变化导致每一层的输入分布会发生改变，进而上层的网络需要不停地去适应这些分布变化，使得我们的模型训练变得困难。上述这一现象叫做Internal Covariate Shift。

<img src="https://upload-images.jianshu.io/upload_images/24408091-0bf76d4a9da28f96.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240" style="zoom:80%;" />

#### 7.6.2 如何减缓Internal Covariate Shift

**（1）白化（Whitening）**

白化（Whitening）是机器学习里面常用的一种规范化数据分布的方法，主要是PCA白化与ZCA白化。白化是对输入数据分布进行变换，进而达到以下两个目的：

- **使得输入特征分布具有相同的均值与方差。**其中PCA白化保证了所有特征分布均值为0，方差为1；而ZCA白化则保证了所有特征分布均值为0，方差相同；
- **去除特征之间的相关性。**

通过白化操作，可以减缓ICS的问题，进而固定每一层网络输入分布，加速网络训练过程的收敛。

然而，白化的方法具有一定缺陷，主要有以下两个问题：

- **白化过程计算成本太高，**并且在每一轮训练中的每一层我们都需要做如此高成本计算的白化操作；
- **白化过程由于改变了网络每一层的分布**，因而改变了网络层中本身数据的表达能力。底层网络学习到的参数信息会被白化操作丢失掉。

**（2）Batch Normalization**

为解决上面两个问题，提出BN方法简化计算过程同时让归一化处理后的数据尽可能保留原始表达能力。
$$
\begin{array}{l}
\mu=\frac{1}{m} \sum_{i} z^{(i)} \\
\sigma^{2}=\frac{1}{m} \sum_{i}\left(z^{(i)}-\mu\right)^{2} \\
z_{\text {norm}}^{(i)}=\frac{z^{(i)}-\mu}{\sqrt{\sigma^{2}+\varepsilon}}
\end{array}
$$
通过上面的变换，**解决了第一个问题，即用更加简化的方式来对数据进行规范化，使得第 $l$层的输入每个特征的分布均值为0，方差为1。**

如同上面提到的，Normalization操作我们虽然缓解了ICS问题，让每一层网络的输入数据分布都变得稳定，但却导致了数据表达能力的缺失。也就是我们通过变换操作改变了原有数据的信息表达（representation ability of the network），使得底层网络学习到的参数信息丢失。另一方面，通过让每一层的输入分布均值为0，方差为1，会使得输入在经过sigmoid或tanh激活函数时，容易陷入非线性激活函数的线性区域。

因此，BN引入的两个可学习（learnable）的参数$\gamma$和$\beta$，恢复数据本身的表达能力，对归一化后端数据进行线性变换，即：
$$
\tilde{z}^{[l]}=\gamma^{[l]} z_{\text {norm}}^{[l]}+\beta^{[l]}
$$
特别地，当$\gamma^{2}=\sigma^{2}, \beta=\mu$时，可以实现等价变换（identity transform）并且保留了原始输入特征的分布信息。

**通过上面的步骤，我们就在一定程度上保证了输入数据的表达能力。**

简单来说,BN减弱了前层参数的作用与后层参数的作用之间的联系，它使得网络每层都可以自己学习，稍稍独立于其它层，这让我们可以使用**更大的学习率**,初值可以更随意，有助于加速整个网络的学习。

#### 7.6.3 BN的效果总结：

**（1）BN使得网络中每层输入数据的分布相对稳定，加速模型学习速度**

BN通过规范化与线性变换使得每一层网络的输入数据的均值与方差都在一定范围内，使得后一层网络不必不断去适应底层网络中输入的变化，从而实现了网络中层与层之间的解耦，允许每一层进行独立学习，有利于提高整个神经网络的学习速度。

**（2）BN使得模型对网络中的参数不那么敏感，简化调参过程，使得网络学习更加稳定**

在神经网络中，我们经常会谨慎地采用一些权重初始化方法（例如Xavier）或者合适的学习率来保证网络稳定训练。

当学习率设置太高时，会使得参数更新步伐过大，容易出现震荡和不收敛。但是使用BN的网络将不会受到参数数值大小的影响。

例如，我们对参数$W$进行缩放得到$aW$。对于缩放前的值$Wu$（$u$表示当前层输入，前一层的输出）,设其均值为$\mu_{1}$，方差为$\sigma_{1}^{2}$；对于缩放值$aWu$，设其均值为$\mu_{2}$，方差为$\sigma_{2}^{2}$，则有：
$$
\mu_{2}=a \mu_{1}, \quad \sigma_{2}^{2}=a^{2} \sigma_{1}^{2}
$$
我们忽略$\epsilon$,则有：
$$
\begin{array}{l}
B N(a W u)=\gamma \cdot \frac{a W u-\mu_{2}}{\sqrt{\sigma_{2}^{2}}}+\beta=\gamma \cdot \frac{a W u-a \mu_{1}}{\sqrt{a^{2} \sigma_{1}^{2}}}+\beta=\gamma \cdot \frac{W u-\mu_{1}}{\sqrt{\sigma_{1}^{2}}}+\beta=B N(W u) \\
\frac{\partial B N((a W) u)}{\partial u}=\gamma \cdot \frac{a W}{\sqrt{\sigma_{2}^{2}}}=\gamma \cdot \frac{a W}{\sqrt{a^{2} \sigma_{1}^{2}}}=\frac{\partial B N(W u)}{\partial u} \\
\frac{\partial B N((a W) u)}{\partial(a W)}=\gamma \cdot \frac{u}{\sqrt{\sigma_{2}^{2}}}=\gamma \cdot \frac{u}{a \sqrt{\sigma_{1}^{2}}}=\frac{1}{a} \cdot \frac{\partial B N(W u)}{\partial W}
\end{array}
$$
可以看到，经过BN操作以后，权重的缩放值会被“抹去”，因此保证了输入数据分布稳定在一定范围内。另外，权重的缩放并不会影响到对$u$的梯度计算；并且当权重越大时，即$a$越大,$\frac{1}{a}$越小，意味着权重$W$的梯度反而越小，这样BN就保证了梯度不会依赖于参数的scale，使得参数的更新处在更加稳定的状态。

因此，在使用Batch Normalization之后，抑制了参数微小变化随着网络层数加深被放大的问题，使得网络对参数大小的适应能力更强，此时我们可以设置较大的学习率而不用过于担心模型divergence的风险。

**（3）BN允许网络使用饱和性激活函数（例如sigmoid，tanh等），缓解梯度消失问题**

在不使用BN层的时候，由于网络的深度与复杂性，很容易使得底层网络变化累积到上层网络中，导致模型的训练很容易进入到激活函数的梯度饱和区；通过normalize操作可以让激活函数的输入数据落在梯度非饱和区，缓解梯度消失的问题；另外通过自适应学习$\gamma$和$\beta$又让数据保留更多的原始信息。

**（4）BN具有一定的正则化效果**

- BN只在每个mini-batch上计算均值和方差，而不是在整个数据集上
- 均值和方差只是由一小部分数据估计得出的，有一些小的噪声，缩放过程从$z^{[l]}$到$\tilde{z}^{[l]}$也有一些噪声，因为它是用有些噪声的均值和方差计算得出的。
- 故和dropout相似，它在每个隐藏层的激活值上增加了噪音。dropout 使一个隐藏的单元以一定的概率乘以 0，以一定的概率乘以 1，所以 dropout含一些噪音，因为它乘以 0 或 1。

因此，类似于dropout，BN有轻微的正则化效果，因为给隐藏单元添加了噪音，使得后部单元不过分依赖任何一个隐藏单元。如果想得到更强大的正则化效果，可以将BN和dropout一起使用。

如果应用较大的mini-batch可以减少噪音，减少正则化效果。

**参考文章**：

[1]: https://zhuanlan.zhihu.com/p/34879333	"Batch Normalization原理与实战"
[2]: BatchNormalizationAcceleratingDeepNetworkTrainingbyReducingInternalCovariateShift



### 7.7 测试时的 Batch Norm

我们知道BN在每一层计算的 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmu) 与 ![[公式]](https://www.zhihu.com/equation?tex=%5Csigma%5E2) 都是基于当前batch中的训练数据，但是这就带来了一个问题：我们在预测阶段，有可能只需要预测一个样本或很少的样本，没有像训练样本中那么多的数据，此时 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmu) 与 ![[公式]](https://www.zhihu.com/equation?tex=%5Csigma%5E2) 的计算一定是有偏估计，这个时候我们该如何进行计算呢？

**论文中的方法**：

利用BN训练好模型后，我们保留了每组mini-batch训练数据在网络中每一层的 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmu_%7Bbatch%7D) 与 ![[公式]](https://www.zhihu.com/equation?tex=%5Csigma%5E2_%7Bbatch%7D) 。此时我们使用整个样本的统计量来对Test数据进行归一化，具体来说使用均值与方差的无偏估计：

![[公式]](https://www.zhihu.com/equation?tex=%5Cmu_%7Btest%7D%3D%5Cmathbb%7BE%7D+%28%5Cmu_%7Bbatch%7D%29)

![[公式]](https://www.zhihu.com/equation?tex=%5Csigma%5E2_%7Btest%7D%3D%5Cfrac%7Bm%7D%7Bm-1%7D%5Cmathbb%7BE%7D%28%5Csigma%5E2_%7Bbatch%7D%29)

得到每个特征的均值与方差的无偏估计后，我们对test数据采用同样的normalization方法：

![[公式]](https://www.zhihu.com/equation?tex=BN%28X_%7Btest%7D%29%3D%5Cgamma%5Ccdot+%5Cfrac%7BX_%7Btest%7D-%5Cmu_%7Btest%7D%7D%7B%5Csqrt%7B%5Csigma%5E2_%7Btest%7D%2B%5Cepsilon%7D%7D%2B%5Cbeta)

**吴恩达老师讲的方法**

对训练阶段每个batch计算的$\mu$和$\sigma^{2}$采用指数加权平均来得到测试阶段$\mu$和$\sigma^{2}$的估计。

我们选择 𝑙层，假设我们有 mini-batch，$X^{\{1\}}$，$X^{\{2\}}$，$X^{\{3\}}$......以及对应的$y$值等等，那么在训练$X^{\{1\}}$时，就得到了$\mu^{\{1\}[l]}$，训练第二个mini-batch时，就会得到$\mu^{\{2\}[l]}$值。然后在这一隐藏层的第三个 mini-batch，得到第三个$\mu$（$\mu^{\{3\}[l]}$）值。使用指数加权平均来追踪这个均值向量的最新平均值，于是该指数加权平均成了你对这一隐藏层的$z$均值的估值。

同样的，可以用指数加权平均来追踪这一层每个mini-batch的$\sigma^{2}$的值。因此在用不同mini-batch训练神经网络的同时，能够得到你所查看每一层$\mu$和$\sigma^{2}$的平均数的实时数值。

最后在测试时对test数据采用同样的normalization方法（同上）。

### 7.8 BN与其他Normalization的方法比较

参考文章：https://zhuanlan.zhihu.com/p/33173246

​                    https://www.cnblogs.com/LXP-Never/p/11566064.html

​                    https://blog.csdn.net/shwan_ma/article/details/85292024

​                    https://bbs.cvmart.net/topics/1569



常用的Normalization方法主要有：Batch Normalization（BN，2015年）、Layer Normalization（LN，2016年）、Instance Normalization（IN，2017年）、Group Normalization（GN，2018年）。它们都是从激活函数的输入来考虑、做文章的，以不同的方式**对激活函数的输入进行 Norm** 的。

将输入的图像shape记为[**N**, **C**hannel, **H**eight, **W**idth]，这几个方法主要的区别就是在，

- **batch Norm**：在batch上，对NHW做归一化，对小batchsize效果不好；
- **layer Norm**：在通道方向上，对CHW归一化，主要对RNN作用明显；
- **instance Norm**：在图像像素上，对HW做归一化，用在风格化迁移；
- **Group Norm**：将channel分组，然后再做归一化；
- **Switchable Norm**：将BN、LN、IN结合，赋予权重，让网络自己去学习归一化层应该使用什么方法。

![img](https://img2018.cnblogs.com/i-beta/1433301/201911/1433301-20191126171358402-795814566.png)

如果把特征图![[公式]](https://www.zhihu.com/equation?tex=x%5Cin%5Cmathbb%7BR%7D%5E%7BN%5Ctimes+C+%5Ctimes+H+%5Ctimes+W%7D)比喻成一摞书，这摞书总共有 N 本，每本有 C 页，每页有 H 行，每行 有W 个字符。

#### 7.8.1 Batch Normalization

$$
\begin{array}{c}
\mu=\frac{1}{m} \sum_{i=1}^{m} x_{i} \\
\sigma=\sqrt{\left.\frac{1}{m} \sum_{i=1}^{m}\left(x_{i}-\mu\right)^{2}\right)} \\
y=\frac{(x-\mu)}{\sqrt{\sigma^{2}+\epsilon}}+\beta=B N(x)
\end{array}
$$

tf.nn.batch_normalization(x, mean, variance, offset, scale, variance_epsilon, name=None`)`

参数

- x：输入数据
- mean：均值
- variance：方差

返回

- 标准化后的数据

**BN缺点:**

- 对batch size的大小比较敏感，由于每次计算均值和方差是在一个batch上，所以如果batch size太小，则计算的均值、方差不足以代表整个数据分布；
- BN实际使用时需要计算并且保存某一层神经网络batch的均值和方差等统计信息，对于对一个固定深度的前向神经网络（DNN，CNN）使用BN，很方便；但对于RNN来说，sequence的长度是不一致的，换句话说RNN的深度不是固定的，不同的time-step需要保存不同的statics特征，可能存在一个特殊sequence比其他sequence长很多，这样training时，计算很麻烦。

#### 7.8.2 Layer Normalization

LN是针对深度网络的某一层的所有神经元的输入按以下公式进行normalize操作。计算该层的平均输入值和输入方差，然后用同一个规范化操作来转换各个维度的输入。
$$
\begin{array}{c}
\mu^{l}=\frac{1}{H} \sum_{i=1}^{H} a_{i}^{l} \\
\sigma^{l}=\sqrt{\frac{1}{H} \sum_{i=1}^{H}\left(a_{i}^{l}-\mu^{l}\right)^{2}}
\end{array}
$$
**BN与LN的区别在于**：

- LN中同层神经元输入拥有相同的均值和方差，不同的输入样本有不同的均值和方差；
- BN中则针对不同神经元输入计算均值和方差，同一个batch中的输入拥有相同的均值和方差。

所以，LN不依赖于batch的大小和输入sequence的深度，因此可以用于batchsize为1和RNN中对边长的输入sequence的normalize操作。

BN 的转换是针对单个神经元可训练的——不同神经元的输入经过再平移和再缩放后分布在不同的区间，而 LN 对于一整层的神经元训练得到同一个转换——所有的输入都在同一个区间范围内。如果不同输入特征不属于相似的类别（比如颜色和大小），那么 LN 的处理可能会降低模型的表达能力。LN用于RNN效果比较明显，但是在CNN上，不如BN。

tf.keras.layers.LayerNormalization(axis=-1, epsilon=0.001, center=True, scale=True)

参数

- axis：想要规范化的轴（通常是特征轴）
- epsilon：将较小的浮点数添加到方差以避免被零除。
- center：如果为True，则将的偏移`beta`量添加到标准化张量。
- `scale`：如果为True，则乘以`gamma`

返回

- shape与输入形状相同的值

#### 7.8.3 Instace Normalization

BN注重对每个batch进行归一化，保证数据分布一致，因为判别模型中结果取决于数据整体分布。

但是图像风格转换中，生成结果主要依赖于某个图像实例，所以对整个batch归一化不适合图像风格化中，因而对HW做归一化。可以加速模型收敛，并且保持每个图像实例之间的独立。
$$
\begin{array}{c}
\mu_{t i}=\frac{1}{H W} \sum_{l=1}^{W} \sum_{m=1}^{H} x_{t i l m} \\
\sigma=\sqrt{\frac{1}{H W}} \sum_{l=1}^{W} \sum_{m=1}^{H}\left(x_{t i l m}-\mu_{y i}\right)^{2} \\
y_{t i j k}=\frac{x_{P t i j k}-\mu_{y i}}{\sqrt{\sigma_{t i}^{2}}-\epsilon}
\end{array}
$$
[`tfa.layers.normalizations.InstanceNormalization`](https://www.tensorflow.org/addons/api_docs/python/tfa/layers/InstanceNormalization)

输入：仅在该层只有一个输入（即，它连接到一个传入层）时适用。

返回：输入张量或输入张量列表。

#### 7.8.4 Group Normalization

主要是针对Batch Normalization对小batchsize效果差，GN将channel方向分group，然后每个group内做归一化，算(C//G)*H*W的均值，这样与batchsize无关，不受其约束。
$$
S_{i}=\left\{k \mid k_{N}=i_{N},\left[\frac{k_{C}}{C / G}\right]=\left[\frac{i_{C}}{C / G}\right]\right\}
$$
Group Normalization 在要求 Batch Size 比较小的场景下或者物体检测／视频分类等应用场景下效果是优于 BN 的。

#### 7.8.5 Switchable Normalization

本篇论文作者认为，

- 第一，归一化虽然提高模型泛化能力，然而归一化层的操作是人工设计的。在实际应用中，解决不同的问题原则上需要设计不同的归一化操作，并没有一个通用的归一化方法能够解决所有应用问题；
- 第二，一个深度神经网络往往包含几十个归一化层，通常这些归一化层都使用同样的归一化操作，因为手工为每一个归一化层设计操作需要进行大量的实验。

因此作者提出自适配归一化方法——Switchable Normalization（SN）来解决上述问题。与强化学习不同，SN使用可微分学习，为一个深度网络中的每一个归一化层确定合适的归一化操作。
$$
\begin{array}{c}
\hat{h}_{h c i j}=\gamma \frac{h_{h c i j}-\sum_{k \in \Omega W_{k} \mu_{k}}}{\sqrt{\sum_{k \in \Omega} w_{k}^{\prime} \sigma_{k}^{2}+\epsilon}}+\beta \\
w_{k}=\frac{e^{\lambda_{k}}}{\sum_{z \in\{i n, l n, b n\} e t}}, \quad k \in\{i n, l n, b n\} \\
\mu_{i n}=\frac{1}{H W} \sum_{i, j}^{H, W} h_{n c i j}, \quad \sigma^{2}=\frac{1}{H W} \sum_{i, j}^{H, W}\left(h_{n c i j}-\mu_{i n}\right)^{2} \\
\mu_{l n}=\frac{1}{C} \sum_{c=1}^{C} \mu_{i n}, \quad \sigma_{l n}^{2}=\frac{1}{C} \sum_{c=1}^{C}\left(\sigma_{i n}^{2}+\mu_{i n}^{2}\right)-\mu_{l n}^{2} \\
\mu_{b n}=\frac{1}{N} \sum_{n=1}^{N} \mu_{i n}, \quad \sigma^{2}=\frac{1}{N} \sum_{n=1}^{N}\left(\sigma_{i n}^{2}+\mu_{i n}^{2}\right)-\mu_{b n}^{2}
\end{array}
$$


#### 7.8.6 weight Normalization

BN 和 LN 均将规范化应用于输入的特征数据 $X$ ，而 WN 则另辟蹊径，将规范化应用于线性变换函数的权重 $W$ ，这就是 WN 名称的来源。

对于人工神经网络中的一个神经元来说，其输出$y$表示为：
$$
y=\phi(\boldsymbol{w} \boldsymbol{x}+b)
$$
其中$w$是$k$维权重向量，$b$是标量偏差，$x$是$k$维输入特征，$\phi(.)$是激活函数。

对权重$w$用参数向量$v$和标量$g$进行表示，则新参数表示为：
$$
\boldsymbol{w}=\frac{g}{\|\boldsymbol{v}\|} \boldsymbol{v}
$$
其中$v$是$k$维向量，$g$是标量，$\|\boldsymbol{v}\|$为$v$的欧式范数。
通过上述参数表示，我们可以发现，$\|\boldsymbol{w}\|=g$与参数$v$独立，而权重$w$的方向也变更为$\frac{v}{\|\boldsymbol{v}\|}$。因此重参数将权重向量$w$用了两个独立的参数表示其幅度和方向。实验证明，在利用SGD优化算法时，重参数加速了网络的收敛速度。

**Gradients:**

则对于$v$,$g$
$$
\begin{array}{c}
\nabla_{g} L=\frac{\nabla_{w} L \cdot \boldsymbol{v}}{\|v\|} \\
\nabla_{v} L=\frac{\boldsymbol{g} \cdot \nabla_{w} L}{\|v\|}-\frac{\boldsymbol{g} \cdot \nabla_{g} L}{\|v\|} \cdot \boldsymbol{v}
\end{array}
$$
其中$\nabla_{w} L$为目标函数对未进行WN的权重为$w$的偏导。

相比与BN，WN带有如下优点：WN的计算量非常低，并且其不会因为mini-batch的随机性而引入噪声统计。在RNN，LSTM，或者Reinforcement Learning上，WN能够表现出比BN更好的性能。



### 7.9 Softmax 回归

Softmax回归：logistic回归的一般形式，将logistic回归推广到 C 类而不仅仅是两类，能在试图识别某一分类时做出预测且任何两个分类之间的决策边界都是线性的。

符号定义：

​	$C$ ：类别总数（类别编号分别为 $0, 2, ..., C-1$），故输出层L的神经元个数为 $C$，每个神经元分别输出该输入属于该类别的概率，$C$ 个神经元输出的概率总和应等于 1 。当 $C=2$ 时，Softmax回归即logistic回归。

#### 7.9.1  Softmax层

​		在分类问题中，神经网络的**输出层**通常选用Softmax激活函数，使得最后输出的是归一化后的实数向量，每一元素代表该样本属于某一类的概率，则该层称为Softmax层。

#### 7.9.2  Softmax激活函数

- 定义：Softmax激活函数只用于多于一个神经元的输出层，它保证所有的输出神经元之和为1，所以一般输出的是小于1的概率值，可以很直观地比较各输出值。

- 作用：为了得到属于每个类别的概率，先通过$e^{z^{[L]}}$将$z^{[L]}$的各元素值映射到 $(0, + \infty)$，然后再归一化到(0, 1)。

- 具体实现：

  在神经网络的最后一层，依然先计算线性部分：
  $$
  z^{[L]}=W^{[L]}a^{[L-1]}+b^{[L]}
  $$
  

  然后应用**Softmax激活函数**：$a^{[L]}=g^{[L]}(z^{[L]})$，该函数的具体计算如下：

  （1）计算临时变量 $t=e^{z^{[L]}}$，对 $z^{[L]}$ 中的所有元素求幂，故 $t$ 的维度与 $z^{[L]}$ 相同。

  （2）对 $t$ **归一化**，得到输出 $a^{[L]}$ 即 $\hat{y}$ ：$a^{[L]}=\frac{e^{z^{[L]}}}{\sum_{j=1}^{n^{[L]}} t_{i}}$，$a^{[L]}$的维度与 $t$ 相同，即也与 $z^{[L]}$ 相同，都为$（n^{[L]}，1）$。其中，对$a^{[L]}$的每一个元素，$a_i^{[L]}=\frac{t_i}{\sum_{j=1}^{n^{[L]}} t_{i}}$。

​       **tips：**之前所学的激活函数如sigmod和ReLU激活函数，都是输入一个实数，输出一个实数，而Softmax激活函数由于需要将所有可能的输出归一化，需要**输入一个向量，输出一个向量**。

#### 7.9.3 softmax配合log似然代价函数训练ANN

​		在人工神经网络（ANN）中，Softmax通常被用作输出层的激活函数。这不仅是因为它的效果好，而且因为它使得ANN的输出值更易于理解。同时，softmax配合log似然代价函数，其训练效果也要比采用二次代价函数的方式好。

​		log似然代价函数的公式为：
$$
C=-\sum_{k} y_{k} \log a_{k}
$$


其中，$a_{k}$表示第k个神经元的输出值，$y_{k}$表示第k个神经元对应的真实值，取值为0或1。

​		该代价函数的简单理解为：在ANN中输入一个样本，那么只有一个神经元对应了该样本的正确类别；若这个神经元输出的概率值越高，则按照以上的代价函数公式，其产生的代价就越小；反之，则产生的代价就越高。

​		为了检验softmax和这个代价函数也可以解决训练ANN时训练速度变慢的问题，接下来的重点就是推导ANN的权重W和偏置b的梯度公式。以偏置b为例：

![img](https://img-blog.csdn.net/20160402220337380)

同理可得：
$$
\frac{\partial C}{\partial w_{j k}}=a_{k}^{L-1}\left(a_{j}^{L}-y_{j}\right)
$$
​		从上述梯度公式可知，softmax函数配合log似然代价函数可以很好地训练ANN，不存在学习速度变慢的问题。

参考文献：https://blog.csdn.net/orangefly0214/article/details/80406584

### 7.10 训练一个 Softmax 分类器

#### 7.10.1 定义损失函数

- 单样本：$$L(\hat{y}, y)=-\sum_{j=1}^C y_{j} \log \hat{y}_{j}$$
- 整个训练集：$J\left(W^{[L]},b^{[L]} \ldots\right)=\frac{1}{m} \sum_{i=1}^{m} L\left(\hat{y}^{(i)}, y^{(i)}\right)$
- 作用：找到样本所属的真实类别，并试图使该类别相应的概率尽可能地高。

#### 7.10.2 正向传播

依次计算出：$z^{[L]} → a^{[L]} = \hat y → L\left(\hat{y}^{(i)}, y^{(i)}\right)$

#### 7.10.3.反向传播（实现梯度下降法）

（1）初始化反向传播：$dz^{[L]}=\frac{\partial J}{\partial z^{[L]}}=\hat y-y$

（2）用深度学习编程框架自动实现反向传播的其余导数计算。

### 7.11 深度学习框架

深度学习框架可以使神经网络的实现变得更简单。每个框架都是针对某一个用户或开发者群体的

#### 7.11.1 现有deep learning框架

- **Caffe/Caffe2**：第一个主流的工业级深度学习工具。开始于2013年底，由UC Berkely的Yangqing Jia老师编写和维护的具有出色的卷积神经网络实现。在计算机视觉领域依然是最流行的工具包，有很多扩展，但是由于一些遗留的架构问题，不够灵活且对递归网络和语言建模的支持很差。
- **CNTK**：是微软研究院开源的深度学习框架。最早由 start the deep learning craze 的演讲人创建，目前已经发展成一个通用的、跨平台的深度学习系统，在语音识别领域的使用尤其广泛。
- **DL4J**：是一个基于 Java 和 Scala 的开源的分布式深度学习库，由 Skymind 于 2014 年 6 月发布，其核心目标是创建一个即插即用的解决方案原型。
- **Keras**：是一个崇尚极简、高度模块化的神经网络库，使用 Python 实现，并可以同时运行在 TensorFlow 和 Theano 上。它旨在让用户进行最快速的原型实验，让想法变为结果的这个过程最短。Theano 和 TensorFlow 的计算图支持更通用的计算，而 Keras 则专精于深度学习。提供了目前为止最方便的 API，用户只需要将高级的模块拼在一起，就可以设计神经网络，它大大降低了编程开销和阅读别人代码时的理解开销。
- **Lasagne**：是一个基于 Theano 的轻量级的神经网络库。Lasagne是 Theano 的上层封装，但又不像 Keras 那样进行了重度的封装，Keras 隐藏了 Theano 中所有的方法和对象，而 Lasagne 则是借用了 Theano 中很多的类，算是介于基础的 Theano 和高度抽象的 Keras 之间的一个轻度封装，简化了操作同时支持比较底层的操作。Lasagne 设计的六个原则是简洁、透明、模块化、实用、聚焦和专注。
- **MXNet**：是李沐和陈天奇等各路英雄豪杰打造的开源深度学习框架，是分布式机器学习通用工具包DMLC的重要组成部分。注重灵活性和效率，文档也非常的详细，同时强调提高内存使用的效率，甚至能在智能手机上运行诸如图像识别等任务。
- **PaddlePaddle**：是百度自主研发的集深度学习核心框架、工具组件和服务平台为一体的技术领先、功能完备的开源深度学习平台，有全面的官方支持的工业级应用模型，涵盖自然语言处理、计算机视觉、推荐引擎等多个领域，并开放多个预训练中文模型。目前已经被中国企业广泛使用，并拥有活跃的开发者社区生态。
- **TensorFlow**：Google开源的其第二代深度学习技术——被使用在Google搜索、图像识别以及邮箱的深度学习框架。是一个理想的RNN（递归神经网络）API和实现，使用了向量运算的符号图方法，使得新网络的指定变得相当容易，支持快速开发。缺点是速度慢，内存占用较大（比如相对于Torch）。
- **Theano**：2008年诞生于蒙特利尔理工学院，主要开发语言是Python。派生出了大量深度学习Python软件包，最著名的包括Blocks和Keras。Theano的最大特点是非常的灵活，适合做学术研究的实验，且对递归网络和语言建模有较好的支持，缺点是速度较慢。
- **Torch**：Facebook力推的深度学习框架，主要开发语言是C和Lua。有较好的灵活性和速度。实现并且优化了基本的计算单元，使用者可以很简单地在此基础上实现自己的算法，不用浪费精力在计算优化上面。核心的计算单元使用C或者cuda做了很好的优化。在此基础之上，使用lua构建了常见的模型。缺点是接口为lua语言，需要一点时间来学习。

参考文献：

- https://blog.csdn.net/dlaicxf/article/details/52846651
- https://www.leiphone.com/news/201702/T5e31Y2ZpeG1ZtaN.html

#### 7.11.2 几种主流dl框架的比较

| **库名称** |  **开发语言**   | **速度** | **灵活性** | **文档** | **适合模型** |  **平台**  | **上手难易** |
| :--------: | :-------------: | :------: | :--------: | -------: | :----------: | :--------: | :----------: |
|   Caffe    |    c++/cuda     |    快    |    一般    |     全面 |     CNN      |  所有系统  |     中等     |
| TensorFlow | c++/cuda/Python |   中等   |     好     |     中等 |   CNN/RNN    | Linux, OSX |      难      |
|   Keras    |     Python      |    快    |     好     |     中等 |   CNN/RNN    | Linux, OSX |      易      |
|   MXNet    |    c++/cuda     |    快    |     好     |     全面 |     CNN      |  所有系统  |     中等     |
|   Torch    |  c++/lua/cuda   |    快    |     好     |     全面 |   CNN/RNN    | Linux, OSX |     中等     |
|   Theano   | Python/c++/cuda |   中等   |     好     |     中等 |   CNN/RNN    | Linux, OSX |      易      |

#### 7.11.3 选择框架的标准

- **便于编程**：包括神经网络的开发和迭代、为产品进行配置等。
- **运行速度**：特别是训练大数据集时，一些框架可以更高效地运行和训练神经网络。
- **框架是否真的开放**：一个真的开放的框架，不仅需要开源，还需要良好的管理。

PyTorch常用代码段合集 https://mp.weixin.qq.com/s/JnIO_HjTrC0DCWtKrkYC8A







