import torch
import matplotlib.pyplot as plt
from PIL import Image
# 指定のクラスをインポート
from ultralytics.models.sam import SAM3SemanticPredictor


def main():
    # 1. 設定の定義 (overrides)
    # modelには "sam3.pt" または "sam3_hiera_l.pt" などのパスを指定します
    overrides = dict(
        conf=0.5,
        task="segment",
        mode="predict",
        model="sam3.pt",
        half=True,
        save=True
    )

    # 画像パスの設定
    image_path2 = "data/farm_insects/Tomato Hornworms/Image_28.jpg"

    # 2. プレディクターの初期化
    print("🚀 Initializing SAM3SemanticPredictor...")
    predictor = SAM3SemanticPredictor(overrides=overrides)

    # 3. 画像のセット
    # predictor.set_image は内部でエンコーダーを走らせ、プロンプトを受け付ける準備をします
    print(f"🖼️ Setting image: {image_path2}")
    predictor.set_image(image_path2)

    # 4. ボックスプロンプトの定義
    # 元のスクリプトの座標を使用 (x1, y1, x2, y2)
    # visual_prompts = [
    #     [104, 213, 240, 707],
    #     [218, 571, 717, 761],
    #     [330, 155, 536, 333],
    #     [483, 148, 921, 417]
    # ]
    visual_prompts = [
        [104, 213, 240, 707]
    ]

    print("📦 Performing semantic inference with SAM 3...")

    # 5. 推論の実行
    # 複数のボックスを渡すことで、それらの概念に基づいたセグメンテーションが行われます
    results = predictor(bboxes=visual_prompts)

    print("✅ Inference completed!")

    # 6. 結果の表示
    # results は Ultralytics の Result オブジェクトのリストです
    for i, result in enumerate(results):
        # プロット用画像を生成
        annotated_image = result.plot()

        # matplotlib で表示
        plt.figure(figsize=(10, 8))
        plt.imshow(annotated_image)
        plt.axis("off")
        plt.title(f"SAM 3 Semantic Segmentation Result {i + 1}")
        plt.show()


def main2():
    # 1. 設定の定義 (overrides)
    # modelには "sam3.pt" または "sam3_hiera_l.pt" などのパスを指定します
    overrides = dict(
        conf=0.5,
        task="segment",
        mode="predict",
        model="sam3.pt",
        half=True,
        save=True
    )

    # 画像パスの設定
    image_path2 = "data/farm_insects/Aphids/Image_19.jpg"

    # 2. プレディクターの初期化
    print("🚀 Initializing SAM3SemanticPredictor...")
    predictor = SAM3SemanticPredictor(overrides=overrides)

    # 3. 画像のセット
    # predictor.set_image は内部でエンコーダーを走らせ、プロンプトを受け付ける準備をします
    print(f"🖼️ Setting image: {image_path2}")
    predictor.set_image(image_path2)

    # 4. ボックスプロンプトの定義
    # 元のスクリプトの座標を使用 (x1, y1, x2, y2)
    # visual_prompts = [
    #     [104, 213, 240, 707],
    #     [218, 571, 717, 761],
    #     [330, 155, 536, 333],
    #     [483, 148, 921, 417]
    # ]
    visual_prompts = [
        [565.5000, 584.5000, 677.0000, 651.0000]
    ]
    results = predictor(text=["green bug"])

    print("📦 Performing semantic inference with SAM 3...")

    # 5. 推論の実行
    # 複数のボックスを渡すことで、それらの概念に基づいたセグメンテーションが行われます
    results2 = predictor(bboxes=results[0].boxes.xyxy[0:1])

    print("✅ Inference completed!")

    # 6. 結果の表示
    # results は Ultralytics の Result オブジェクトのリストです
    for i, result in enumerate(results):
        # プロット用画像を生成
        annotated_image = result.plot(labels=False)

        # matplotlib で表示
        plt.figure(figsize=(10, 8))
        plt.imshow(annotated_image)
        plt.axis("off")
        plt.title(f"SAM 3 Semantic Segmentation Result {i + 1}")
        plt.show()

    for i, result in enumerate(results2):
        # プロット用画像を生成
        annotated_image = result.plot(labels=False)

        # matplotlib で表示
        plt.figure(figsize=(10, 8))
        plt.imshow(annotated_image)
        plt.axis("off")
        plt.title(f"SAM 3 Semantic Segmentation Result {i + 1}")
        plt.show()


def main3():
    overrides = dict(
        conf=0.5,
        task="segment",
        mode="predict",
        model="sam3.pt",
        half=True,
        save=True
    )

    # 画像パスの設定
    image_path2 = "data/farm_insects/Aphids/Image_19.jpg"

    # 2. プレディクターの初期化
    print("🚀 Initializing SAM3SemanticPredictor...")
    predictor = SAM3SemanticPredictor(overrides=overrides)

    # 3. 画像のセット
    # predictor.set_image は内部でエンコーダーを走らせ、プロンプトを受け付ける準備をします
    print(f"🖼️ Setting image: {image_path2}")
    predictor.set_image(image_path2)

    # 4. ボックスプロンプトの定義
    # 元のスクリプトの座標を使用 (x1, y1, x2, y2)
    # visual_prompts = [
    #     [104, 213, 240, 707],
    #     [218, 571, 717, 761],
    #     [330, 155, 536, 333],
    #     [483, 148, 921, 417]
    # ]
    visual_prompts = [
        [565.5000, 584.5000, 677.0000, 651.0000]
    ]
    results = predictor(text=["green bug"])

    predictor.inference_features(predictor.features, src_shape=src_shape, text=["person"])

if __name__ == "__main__":
    main()