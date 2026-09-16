# Iris Completed Mass LPIPS

実LPIPS推論を行うバックエンド付きPWAです。

## 判定対象
入力画像そのものではなく、バックエンドで
1. 正方形化
2. 円形虹彩領域のみ保持
3. grayscale
4. sigma=16/32/48 のGaussian low-pass
5. RGB 3chへ複製
を行い、各スケールのLPIPS(AlexNet) distanceを返します。

### 重要
LPIPSは距離です。0に近いほど近いという性質はありますが、
LPIPS値から「強い類似」「極めた類似」へ変換する公式境界はありません。
この版では恣意的な100点換算・言語帯換算を入れていません。

## 起動
Docker:
    docker build -t iris-lpips .
    docker run --rm -p 8000:8000 iris-lpips

ブラウザ:
    http://localhost:8000

無料/有料のDocker対応ホスティングへ置けばHTTPS化できます。
iPhone Safariで開き「ホーム画面に追加」するとPWAとして使えます。

## 注意
入力は、できるだけ虹彩円を正方形の中央へ合わせた画像にしてください。
写真条件差をさらに抑えるには、次版で円位置・半径の手動調整UIを追加するのが適切です。


## v2
画像A/Bごとに虹彩中心X/Yと半径を手動調整できます。サーバー側でその円を切り出して同一径化した後にLPIPSを実行します。
