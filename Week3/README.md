# Week3
---
### task 01
rewrite linear regression in python way 

### task 02
finish logstic regression in python way

### task 03
A person is swimming across several rivers.

Speeds of those rivers are different: v1, v2, ..., vn. To simplify this problem, we only consider the speed in vertical direction.

The person’s speed is v. It’s a constant, no way to change that. And the angle of the person’s velocity to horizontal line is a1, a2, ..., an.

The total time for swimming is T. And, **the person must pass those rivers.**



**task is:**

Find out an equation to determine by choosing what angles (a1, a2, ..., an) the person can get maximum distance in vertical direction (That is to say, please maximize *dh* by determining *a1, a2, ...**, an*) under the total time *T*. 【You are not required to give out concrete angle numbers, a “cost function” that can bederived from is enough】

Tips: For this question, a mathematical tool you may need is called “Lagrangian Multiplier”. Which mean![]()s,when you provide a formula, say *E*, which still need to satisfy some more conditions, say *a > 1*, for the convenience of calculating, we can write those 2 parts (formula *E* and condition *a > 1*) together as one new formula. Here the new formula will be: E – λ(a - 1).

![page1image59899904.jpg](https://raw.githubusercontent.com/sszove/CV_CNN/master/Week3/README.assets/page1image59899904-4902012.jpg) 



**Solution**

x asix -> river land

Y asix -> across river

$vx = v*sin(a)$

$vy = v*cos(a)$

$h = vx * t$

$s = vy*t$

w--> river width

$L(a,t,lambda) = h - lambda*(s - w)$



Ref link:[拉格朗日乘子法](https://www.cnblogs.com/wt869054461/p/6760200.html)

Ref link:[图解拉格朗日乘子法](https://blog.csdn.net/ccnt_2012/article/details/81326626)

