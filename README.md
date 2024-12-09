# 序論
## 1.1. 研究の背景
### 1.1.1. 感染症の定義と分類、および細菌感染症
### 1.1.2. 細菌の生物学的特性と人間社会への影響
### 1.1.3. 敗血症の病態生理
### 1.1.4. 臨床現場における感染症診療の現状

## 1.2. 研究の動機
### 1.2.1. 現場における問題点
### 1.2.2. 医師主導の医工連携研究の意義

## 1.3. 研究の目的と方法
### 1.3.1. 自家蛍光分析と機械学習を用いた細菌同定システムの研究開発
### 1.3.2. 実用化に向けた開発戦略と事業化計画

## 1.4. 論文の構成

# 細菌の自家蛍光の計測
## 2.1. はじめに
### 2.1.1. 有機物の測定手法の概要と分光計測の位置づけ
### 2.1.2. 蛍光分光法の原理と特徴
### 2.1.3. 自家蛍光の原理と特徴、菌種同定における利用
### 2.1.4. 蛍光分光光度計の構造と特徴

## 2.2. 細菌サンプルの準備
### 2.2.1. 菌株の選定と培養
### 2.2.2. サンプルの調製
### 2.2.3. 菌種毎の検量線の作成

## 2.3. 蛍光分光光度計によるサンプルの計測
### 2.3.1. 実験セットアップ
### 2.3.2. データ収集、前処理
### 2.3.3. 細菌自家蛍光の全体的な傾向

## 2.4. 自家蛍光の菌種間の差異
### 2.4.1. 菌株ごとのEEMの比較
### 2.4.2. EEMから抽出した個別の励起波長の蛍光スペクトルによる比較
### 2.4.3. 統計学的な解析を用いた比較

## 2.5. まとめ

# 機械学習による自家蛍光データを用いた菌種同定
## 3.1. はじめに
### 3.1.1. 機械学習について
### 3.1.2. 細菌の自家蛍光データに対する機械学習の適用について
### 3.1.3. ベイズ最適化について

## 3.2. 菌自家蛍光スペクトルデータに対する機械学習の適用
### 3.2.1. データの前処理
### 3.2.2. 機械学習モデルの選択と構築
### 3.2.3. モデルのトレーニングと評価方法
### 3.2.4. チューニングによる診断精度の変化

## 3.3. 励起・蛍光波長の診断精度への影響
### 3.3.1. 単波長のスペクトルによる診断精度について
### 3.3.2. 2波長のスペクトルによる診断精度について
### 3.3.3. 3波長のスペクトルによる診断精度について
### 3.3.4. 4波長以上のスペクトルによる診断精度について

## 3.4. 波長幅の診断精度への影響
### 3.4.1. 励起光バンド幅の変化によるスペクトル形状の変化と、信号積算によるシミュレーション
### 3.4.2. 合成蛍光スペクトルの波長領域およびその個数と診断精度について
### 3.4.3. 合成励起スペクトルの波長領域およびその個数と診断精度について

## 3.5. まとめ

# 菌種の同定に最適化した計測装置の開発
## 4.1. 光源の選定と最適化
### 4.1.1. 試作計測装置用LEDの選定と、発光特性を反映した合成蛍光スペクトルのシミュレーション
### 4.1.2. 菌種診断における計測装置の励起光に使用するLEDの、ベイズ最適化を使った最適化

## 4.2. 光源と光学系の開発
### 4.2.1. 単一LEDによる試作
### 4.2.2. 単一LEDによる試作を用いた測定結果
### 4.2.3. 複数LEDによる試作

## 4.3. 検出系の開発
### 4.3.1. 分光計のセンサ温度による暗電流ノイズ
### 4.3.2. 分光計の設計と試作
### 4.3.3. 試作した分光計による計測と性能評価
### 4.3.4. 微弱光計測のための分光計の設計と試作
### 4.3.5. 試作した分光計による計測
### 4.3.6. センサの設計と試作
### 4.3.7. 試作したセンサの駆動と計測

## 4.4. まとめ

# 事業実践と起業に向けた環境構築
## 5.1. 市場分析
### 5.1.1. 細菌迅速検査市場の概要
### 5.1.2. 市場動向と成長要因
### 5.1.3. 競合企業
### 5.1.4. 市場ニーズ、顧客要求、市場参入の機会とそれに伴うリスク

## 5.2. 競合製品の現状と新アプローチによる優位性の検討
### 5.2.1. 現状の市場における競合製品の概要、それぞれの利点と課題、および臨床現場における評価
### 5.2.2. 新しいアプローチによる競争優位性

## 5.3. ビジネスプランの戦略的構築と実行計画
### 5.3.1. ビジネスモデルの構築
### 5.3.2. 収益予測
### 5.3.3. マイルストーンの設定
### 5.3.4. パートナーシップの構築と成功指標の設定

## 5.4. 医療機器市場参入のための法的要件とリスク管理
### 5.4.1. 医療機器に関する各国の法規制の概要
### 5.4.2. 本邦における医療機器承認プロセスと認証プロセス
### 5.4.3. 新規参入の課題とリスク管理
### 5.4.4. 国際規格と標準への適合

## 5.5. 研究開発費の確保と持続的資金戦略
### 5.5.1. 研究開発費の必要性と目的
### 5.5.2. 資金調達の戦略
### 5.5.3. 助成金・補助金と他人資本との違い
### 5.5.4. 学術領域とビジネス領域の求められるコミュニケーションの違い

## 5.6. 新規事業創出に向けた支援リソースの活用
### 5.6.1. 起業家を支援する様々な制度
### 5.6.2. 経済産業省の「始動」プログラム
### 5.6.3. JETROのドイツ出展&ビジネスアテンド支援事業
### 5.6.4. MEDISOの医療系ベンチャー・トータルサポート事業
### 5.6.5. 都道府県の特定創業支援事業

## 5.7. まとめ

# 結論
## 6.1. 研究の総括
## 6.2. 結論
## 6.3. 研究の限界と今後の展望

# 補遺
## 7.1. データセット
## 7.2. コード
## 7.3. 使用部品一覧
### 7.3.1. LED光源の検討時のシミュレーションに使用したLED(4.1.参照)
### 7.3.2. 単一LEDによる試作に使用した部品
### 7.3.3. 単一LEDによる試作に使用した部品
### 7.3.4. 分光計の試作、評価に使用した部品
### 7.3.5. 微弱光用の分光計の試作に使用した部品
### 7.3.6. センサの試作、評価に使用した部品
