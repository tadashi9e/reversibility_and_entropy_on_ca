# -*- mode:python;coding:utf-8 -*-
import argparse
import random
import sys
import cupy as cp
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

u'''
# 「ラングトンのアリ」を格子ガス風に GPU 上で動かす

## はじめに

[ラングトンのアリ](https://ja.wikipedia.org/wiki/%E3%83%A9%E3%83%B3%E3%82%B0%E3%83%88%E3%83%B3%E3%81%AE%E3%82%A2%E3%83%AA)
は、二次元チューリングマシンであることが知られています。

複数のアリが動く様子を GPU(cupy) 上で高速に動作させたいと思います。

ラングトンのアリに目をつけたのは、

「ミクロな世界を記述するであろう可逆な法則、例えば
ニュートン力学、
シュレディンガー方程式、
電磁方程式
などの組み合わせからなるはずのマクロな世界が不可逆に見えるのはなぜなのか」

という疑問がきっかけです。

「極度に単純化された可逆セル・オートマトンの世界でもエントロピーが増大することを示せば、
基礎方程式の可逆性とエントロピー増大とは無関係であることが示せるに違いない」
というのが大本の発想で、
さらに、可逆セル・オートマトンの一つとして格子ガスが使えるのではないか、
さらに物理的アナロジーを無視してラングトンのアリでもよいのではないか、
ということで、
自由度の大きい系を人間が観測可能な時間内で実行するために
GPU 上での大規模なシミュレーションを可能にしたいと思いました。

## 格子ガスからラングトンのアリへ

### セルへの情報のエンコード

格子ガスセル・オートマトンでは、
各セルの各ビットに粒子の運動量・粒子数の情報を持たせます。

- そのセルに停止している粒子の有無
- 上方向に動いている粒子の有無
- 右方向に動いている粒子の有無
- 下方向に動いている粒子の有無
- 左方向に動いている粒子の有無

といった具合です
（実際には正方格子上の４方向だと不自然な動きになるので、
六角格子上の６方向に動く形にしたりしますが、概略こんな感じです）。

ラングトンのアリも同様に、各セルのビットに
当該セルが白黒二色のどちらに塗られているかいう情報とは別に、
アリの有無・移動方向の情報を持たせることにします。

- N ビット : 当該セルに北を向いているアリがいるかどうかを表すビット
- E ビット : 当該セルに東を向いているアリがいるかどうかを表すビット
- S ビット : 当該セルに南を向いているアリがいるかどうかを表すビット
- W ビット : 当該セルに西を向いているアリがいるかどうかを表すビット
- BW ビット : 当該セルが白(0)であるか、黒(1)であるかを表すビット

~~~python
'''
BIT_N = cp.uint8(1 << 0)
BIT_E = cp.uint8(1 << 1)
BIT_S = cp.uint8(1 << 2)
BIT_W = cp.uint8(1 << 3)
BIT_BW = cp.uint8(1 << 4)

u'''
~~~

当該セルにアリがいるかどうか
（当該セルの白黒を反転する必要があるか）を判定するために、
以下のようなビットマスクを用意しておくと便利です:

~~~python
'''
BITS_NEWS = 0x0f
u'''
~~~

### 注意点

注意点 1

この方法だと同じセルに同じ向きのアリが同時に複数存在できません。
しかし、この制限が問題になることはありえません。
なぜなら、同じセルに同じ向きのありが同時に複数存在したとして、それらのアリは
それまでもそれからもずっと同じ場所を動き続けることになるからです。
逆に言うならば、別のセルや別の向きにいた複数のアリが
なにかのタイミングで同じセルの同じ向きに入り込んでしまう、
ということもありえません。

すべてのアリは無個性であり、すべてのアリはステップごとに
一斉に同じタイミングで向きを変え、セルの色を変え、前進する、
という前提に立つ限りはこのように問題は起きません。

注意点 2

シミュレーションの途中で、アリの総数は一定である必要があります。

一方で、各セルに情報を持たせる、というやりかただとアリの総数が一定であることは
自明でありません。

セルの更新ルールを正しく決めることでアリが増減しないようにします。

### アリを動かす

格子ガスのシミュレーションでは、一ステップの動作を

- 粒子の衝突・散乱
- 粒子の並進

のスモールステップに分けて実装できます。

同様に、アリの動きも

1. アリがセルの色を見て向きを変える
2. アリがセルの色を変える
3. アリが一つ進む

のスモールステップに分解することができます。

ここでは、「アリが向きを変える」と「アリがセルの色を変える」をひとまとめにして

1. アリがセルの色を見て向きを変えるのと同時に、セルの色を変える
2. アリが一つ進む

の２つのスモールステップとして実装することにしました。

#### アリがセルの色を見て向きを変えるのと同時に、セルの色を変える処理

セルのビットを見て更新していくことで実装します。

1. セルの色が白である（BW ビットが 0 である場合）なら時計方向にアリの向きを変える、黒である（BW ビットが 0 でない場合）なら反時計方向にアリの向きを変える
2. セルの色を反転させる（BW ビットの値を反転する）

~~~python
'''
def rotate_and_flip(field : cp.ndarray) -> cp.ndarray:
    cw = (
        ((field & BIT_N) != 0).astype(cp.uint8) * BIT_E |
        ((field & BIT_E) != 0).astype(cp.uint8) * BIT_S |
        ((field & BIT_S) != 0).astype(cp.uint8) * BIT_W |
        ((field & BIT_W) != 0).astype(cp.uint8) * BIT_N)
    ccw = (
        ((field & BIT_N) != 0).astype(cp.uint8) * BIT_W |
        ((field & BIT_E) != 0).astype(cp.uint8) * BIT_N |
        ((field & BIT_S) != 0).astype(cp.uint8) * BIT_E |
        ((field & BIT_W) != 0).astype(cp.uint8) * BIT_S)
    rotated = cp.where((field & BIT_BW) == 0, cw, ccw)
    flip = ((field & BITS_NEWS) != 0).astype(cp.uint8) * BIT_BW
    flipped = (field & BIT_BW) ^ flip
    return rotated | flipped
u'''
~~~

ここで、

~~~python
((field & BIT_N) != 0).astype(cp.uint8) * BIT_E
~~~

の部分は、セルのビット N が ON だった場合にはビット E を ON にする、という意味です。

条件分岐やループを用いずに、なるべく GPU 上で処理しようと、このような書き方をしています。

さらに、各ビットの位置が決まっていることを前提としてビットシフトで書くことも可能なのですが、
将来、ビットの定義を変えたくなったときのために `.astype(cp.uint8)` を使った形で
処理を書いています。

#### アリが一つ進む処理

cupy(numpy) の `roll` を使ってアリを進めることができます:

~~~python
'''
def forward(field : cp.ndarray) -> cp.ndarray:
    return (
        ((cp.roll(field, -1, axis = 0) & BIT_N) != 0).astype(cp.uint8) * BIT_N |
        ((cp.roll(field, 1, axis = 1) & BIT_E) != 0).astype(cp.uint8) * BIT_E |
        ((cp.roll(field, 1, axis = 0) & BIT_S) != 0).astype(cp.uint8) * BIT_S |
        ((cp.roll(field, -1, axis = 1) & BIT_W) != 0).astype(cp.uint8) * BIT_W |
        cp.bitwise_and(field, BIT_BW))
u'''
~~~

`roll` は周期的境界条件のもとでの回転操作なので、壁面でアリが消えることはありません
（領域の端まで進んだアリは反対側の端に現れる）。

#### アリが一つ戻る処理

ラングトンのアリの動作を時間反転させたときの動作も見たいので、アリが戻る処理も実装しました。

アリを進める処理をそのまま逆方向に行うだけです:

~~~python
'''
def rev_forward(field : cp.ndarray) -> cp.ndarray:
    return (
        ((cp.roll(field, 1, axis = 0) & BIT_N) != 0).astype(cp.uint8) * BIT_N |
        ((cp.roll(field, -1, axis = 1) & BIT_E) != 0).astype(cp.uint8) * BIT_E |
        ((cp.roll(field, -1, axis = 0) & BIT_S) != 0).astype(cp.uint8) * BIT_S |
        ((cp.roll(field, 1, axis = 1) & BIT_W) != 0).astype(cp.uint8) * BIT_W |
        cp.bitwise_and(field, BIT_BW))
u'''
~~~

「アリがセルの色を見て向きを変えるのと同時に、セルの色を変える処理」の方は
時間反転の場合にもまったく同じ処理が使えるので、こちらは改めて実装する必要はありません。

このことは、rotate_and_flip を二回実行するともとに戻ることから確認できます。

#### １ステップの動き

というわけで、

1. アリがセルの色を見て向きを変えるのと同時に、セルの色を変える
2. アリが一つ進む

という２つのスモールステップを組み合わせて、以下のように１ステップの処理を実装できます:

~~~python
'''
def update_field(field : cp.ndarray) -> None:
    field2 = forward(rotate_and_flip(field))
    cp.copyto(field, field2, casting = 'safe')
u'''
~~~

時間反転についても同様に、以下のような実装になります:

~~~python
'''
def rev_update_field(field : cp.ndarray) -> None:
    field2 = rotate_and_flip(rev_forward(field))
    cp.copyto(field, field2, casting = 'safe')

u'''
~~~

## エントロピーの計算方法

エントロピーをどう定義するかは、なにを見たいかによって変わります。

今回は、
可逆セル・オートマトンにおいても
時間を進めると全体としてマクロなエントロピーが増大していく
（逆に時間反転すると減少していく）ということを示したかったので、

```math
H = - ( p_b log_2 p_b + p_w log_2 p_w )
```

としました（ここで $p_b$ は黒いセルの割合、$p_w$ は白いセルの割合）:

~~~python
'''
def calc_entropy_bw(field : cp.ndarray) -> float:
    ncells = field.size
    c_b = cp.sum((field & BIT_BW) != 0)
    p_b = c_b / ncells
    p_w = 1.0 - p_b
    entropy = -(cp.where(p_b > 0, p_b * cp.log2(p_b), 0) +
                cp.where(p_w > 0, p_w * cp.log2(p_w), 0))
    return float(entropy)
u'''
~~~

全体が白一色あるいは黒一色となる可能性も考えて、念の為に `cp_where` での分岐も入れてあります。

ここで定義したエントロピーは、
アリの配置というミクロ状態そのものではなく、
セルの白黒分布というマクロ変数に対する粗視化エントロピーです。

## メイン処理など

ステップごとの処理は書いてしまったので、あとはこれを繰り返し実行するだけです。

正しく可逆セル・オートマトンとして実装できていて、
逆ルールを適用することで時間逆転できる、
ということを示すために、
一定ステップ後（デフォルトでは 1000 ステップ後）から逆ルールを適用するようにしています。

エントロピーの推移を含めて、MP4 としてアニメーション生成する機能も付けました。

~~~python
'''
def update(count : int, max_count : int, reverse_count : int,
           field : cp.ndarray,
           img : matplotlib.image.AxesImage,
           line : matplotlib.lines.Line2D,
           ax2 : plt.axis,
           entropy_sequence_bw : list[float]) -> list[matplotlib.artist.Artist]:
    if count % 10 == 0:
        sys.stdout.write('\r{} / {}'.format(count, max_count))
    if reverse_count >= 0 and count < reverse_count:
        update_field(field)
    else:
        rev_update_field(field)
    if count == reverse_count:
        ax2.axvline(reverse_count, color = 'r')
    img.set_data(cp.asnumpy(field))
    entropy_bw = calc_entropy_bw(field)
    entropy_sequence_bw[count] = entropy_bw
    line.set_data(range(count + 1),
                  entropy_sequence_bw[:count + 1])
    ax2.set_xlim(0, count + 1)
    ax2.set_ylim(0, 1)
    return [img, line]
def generate_animation(
        field : cp.ndarray,
        max_count : int,
        reverse_count : int,
        target : str) -> None:
    print('generating animation...')
    img = ax1.imshow(cp.asnumpy(field), cmap='Greys', interpolation='nearest')
    (line,) = ax2.plot([], [])
    entropy_sequence_bw = [0.0] * max_count
    a = animation.FuncAnimation(
        fig,
        update,
        fargs = (max_count, reverse_count, field,
                 img, line, ax2, entropy_sequence_bw),
        interval = 1,
        blit = True,
        frames = max_count,
        repeat = False)
    print('saving...')
    a.save(target, writer = "ffmpeg")
    print(f'\r{max_count}')

def simulate(field : cp.ndarray,
             max_count : int,
             reverse_count : int,
             ax1 : plt.axis,
             ax2 : plt.axis) -> None:
    img = ax1.imshow(cp.asnumpy(field), cmap='Greys', interpolation='nearest')
    (line,) = ax2.plot([], [])
    entropy_sequence_bw = [0.0] * max_count
    for count in range(max_count):
        update(count, max_count, reverse_count, field,
               img, line, ax2, entropy_sequence_bw)
        plt.pause(0.001)
    print(f'\r{max_count}')
    plt.show()

parser = argparse.ArgumentParser()
parser.add_argument("-W", "--width",
                    type = int,
                    default = 300,
                    help = "Field width (default 300)")
parser.add_argument("-H", "--height",
                    type = int,
                    default = 300,
                    help = "Field height (height 150)")
parser.add_argument("-a", "--animation",
                    type = str,
                    help = "Generate mp4 animation (ex. animation.mp4)")
parser.add_argument("-c", "--count",
                    type = int,
                    default = 3000,
                    help = "Max step count (default 3000)")
parser.add_argument("-r", "--reverse",
                    type = int,
                    default = 1000,
                    help = "Reversing start count (default 1000, negative means NO REVERSE)")
parser.add_argument("-s", "--seed",
                    type = int,
                    default = 8,
                    help = "Random seed")
args = parser.parse_args()
random.seed(args.seed)

field = cp.zeros((args.height, args.width), dtype = cp.uint8)
for n in range(1, 50):
    y = random.randint(0, args.height - 1)
    x = random.randint(0, args.width - 1)
    field[y, x] = 1 << random.randint(0, 3)
fig, (ax1, ax2) = plt.subplots(
    2,1, gridspec_kw={'height_ratios': [5, 1]})
ax1.axis('off')

if args.animation:
    generate_animation(field, args.count, args.reverse, args.animation)
else:
    simulate(field, args.count, args.reverse, ax1, ax2)
print('done')
u'''
~~~

# まとめ

格子ガス風の流儀で、GPU(cupy) 上で動く「ラングトンのアリ」を実装しました。

[ほぼ同じ方針でGPU.js 上で実装したラングトンのアリ](https://tadashi9e.github.io/gpu_js_ca/langtons-ant.html)
も以前作成しましたが、このときにはエントロピーについての問題意識は持っていませんでした。

なので、エントロピーの計算や時間反転も実験できるようにしてみました。

今回の実装により、

- 完全に可逆なルールのもとであっても、マクロな観測量は不可逆的に見えるということ、
- 時間反転させた場合にエントロピーが減少していくこと、
- 初期状態を超えて時間反転を継続していくと逆にエントロピーは増大していくこと

などを実験的に確認することができました。
'''
