# 磨损体积计算公式

## 1. 原始公式 (1-30号刀具，43，44号刀具)

### 正向：半径 → 体积

$$V = 2\pi \Delta R \cdot N \left[ TR - \frac{T}{2}\Delta R + R\tan(\theta)\Delta R - \frac{2}{3}\tan(\theta)(\Delta R)^2 \right]$$

其中：
- $V$: 磨损体积 (mm³)
- $\Delta R$: 磨损半径 (mm)
- $T$: 刀具厚度 (mm), 默认 25.4
- $R$: 刀具位置半径 (mm), 默认 241.3
- $\theta$: 刀具角度 (度), 默认 20°
- $N$: 刀具数量, 默认 1

### 反向：体积 → 半径

求解三次方程：

$$-\frac{2}{3}\tan(\theta)(\Delta R)^3 + \left[R\tan(\theta) - \frac{T}{2}\right](\Delta R)^2 + TR \cdot \Delta R - \frac{V}{2\pi N} = 0$$

**解法**：选择最小的正实根作为物理解

---

## 2. 新公式 (31-42号刀具)

### 参数定义

- $T = 25.4$ mm
- $H = 2413$ mm (刀具高度)
- $\theta = 20°$
- $d = (7-30) \times \frac{70}{13}$ (度)

### 中间变量计算

#### 计算 $h_0$:

$$h_0 = \frac{\left[\frac{T}{2} + \frac{R}{\tan(d)}\right] \cdot \tan(90°-\theta)}{\tan(90°- \theta) - \tan(d)}$$

#### 计算 $h_1$:

$$h_1 = \left[\frac{R}{\tan(d)} - \frac{T}{2}\right] \cdot \left[1 - \frac{\sin(d) \cdot \cos\theta}{\sin(90°+\theta - d)}\right] \cdot\tan(d)$$

### 判定条件

当 $h_0 = 2R$ 时，计算判定值：

$$\text{condition} = \frac{h_0}{\tan(d)} - \frac{h_0}{\tan(90°- \theta)}$$

**判定规则**：
- 若 $\text{condition} > T_{pd}$ (其中 $T_{pd} = T$)，使用**公式1**
- 否则，使用**公式2**
$$V=\pi\cdot\text{condition}\cdot\ h_0 \left[H-\frac{h_0}{3}\right]$$

### 公式1：$V = V_0 - V_1$

$$V_0 = \pi \left[\frac{R}{\tan(d)} + \frac{T}{2}\right] \cdot h_0 \cdot \left[H - \frac{h_0}{3}\right]$$

$$V_1 = \pi \left[\frac{R}{\tan(d)} - \frac{T}{2}\right] \cdot h_1 \cdot \left[H - \frac{h_1}{3}\right]$$

$$V = V_0 - V_1$$

### 公式2：
$$V=\pi\cdot\text{condition}\cdot\ h_0 \left[H-\frac{h_0}{3}\right]$$



---

## 3. 特殊规则

### 44号刀具
44号刀具的磨损量与43号刀具完全一致：

$$V_{44} = V_{43}$$

44号刀具和43号刀具的磨损公式使用原始公式计算计算

---
