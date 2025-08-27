import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# ----------------------------
# 1. 추천 고정 매핑 (9개 그룹)
# ----------------------------
recommendations = {
    # 그룹 1: 26SS_01, 03, 10, 14, 16, 17, 20, 23
    "26SS_01": [
        {"Thread_No": 19203, "Thread_Name": "WHALE BLUE", "R": 0, "G": 51, "B": 102},
        {"Thread_No": 13673, "Thread_Name": "FRENCH NAVY", "R": 10, "G": 35, "B": 85},
        {"Thread_No": 13685, "Thread_Name": "INKY NAVY", "R": 15, "G": 30, "B": 60},
        {"Thread_No": 13682, "Thread_Name": "SEPIA NAVY", "R": 25, "G": 45, "B": 90},
    ],
    "26SS_03": [...], "26SS_10": [...], "26SS_14": [...],
    "26SS_16": [...], "26SS_17": [...], "26SS_20": [...], "26SS_23": [...],

    # 그룹 2: 26SS_02
    "26SS_02": [
        {"Thread_No": 16402, "Thread_Name": "BEIGE", "R": 210, "G": 190, "B": 150},
        {"Thread_No": 16400, "Thread_Name": "LIGHT CAMEL", "R": 200, "G": 170, "B": 130},
        {"Thread_No": 19209, "Thread_Name": "CAMEL", "R": 190, "G": 150, "B": 100},
        {"Thread_No": 16398, "Thread_Name": "HOT SAND", "R": 230, "G": 200, "B": 160},
    ],

    # 그룹 3: 26SS_04, 05
    "26SS_04": [
        {"Thread_No": 16363, "Thread_Name": "PEARL", "R": 245, "G": 245, "B": 240},
        {"Thread_No": 16362, "Thread_Name": "CREAM WHITE", "R": 250, "G": 250, "B": 245},
        {"Thread_No": 16303, "Thread_Name": "SWISS CREAM", "R": 240, "G": 235, "B": 220},
        {"Thread_No": 16302, "Thread_Name": "FROSTED BEIGE", "R": 230, "G": 225, "B": 210},
    ],
    "26SS_05": [...],

    # 그룹 4: 26SS_06
    "26SS_06": [
        {"Thread_No": 16137, "Thread_Name": "BRILLIANT ORANGE", "R": 240, "G": 90, "B": 40},
        {"Thread_No": 16326, "Thread_Name": "SPICED COPPER", "R": 200, "G": 80, "B": 40},
        {"Thread_No": 16329, "Thread_Name": "MAPLE TAN", "R": 180, "G": 90, "B": 60},
        {"Thread_No": 16139, "Thread_Name": "VIVID ORANGE", "R": 255, "G": 100, "B": 50},
    ],

    # 그룹 5: 26SS_07, 11, 13
    "26SS_07": [
        {"Thread_No": 16481, "Thread_Name": "BABY BLUE", "R": 180, "G": 220, "B": 240},
        {"Thread_No": 16276, "Thread_Name": "PALE TURQUOISE", "R": 160, "G": 210, "B": 230},
        {"Thread_No": 16482, "Thread_Name": "TAHITI BLUE", "R": 140, "G": 190, "B": 220},
        {"Thread_No": 15547, "Thread_Name": "PALE BLUE", "R": 200, "G": 230, "B": 250},
    ],
    "26SS_11": [...], "26SS_13": [...],

    # 그룹 6: 26SS_08, 09, 12
    "26SS_08": [
        {"Thread_No": 14846, "Thread_Name": "ELECTRIC GREY", "R": 140, "G": 140, "B": 140},
        {"Thread_No": 16292, "Thread_Name": "SECRET BLUE", "R": 18, "G": 69, "B": 139},
        {"Thread_No": 13643, "Thread_Name": "MAGIC BLUE", "R": 25, "G": 77, "B": 142},
        {"Thread_No": 16296, "Thread_Name": "LOYAL BLUE", "R": 33, "G": 83, "B": 145},
    ],
    "26SS_09": [...], "26SS_12": [...],

    # 그룹 7: 26SS_15, 18, 21, 24
    "26SS_15": [
        {"Thread_No": 16151, "Thread_Name": "JAZZY VIOLET", "R": 130, "G": 70, "B": 160},
        {"Thread_No": 13822, "Thread_Name": "PINK DREAM", "R": 220, "G": 160, "B": 200},
        {"Thread_No": 16016, "Thread_Name": "SOFT ROSE", "R": 240, "G": 180, "B": 200},
        {"Thread_No": 16019, "Thread_Name": "ETERNAL PINK", "R": 210, "G": 140, "B": 170},
    ],
    "26SS_18": [...], "26SS_21": [...], "26SS_24": [...],

    # 그룹 8: 26SS_19
    "26SS_19": [
        {"Thread_No": 16043, "Thread_Name": "MAGENTA", "R": 200, "G": 40, "B": 120},
        {"Thread_No": 16157, "Thread_Name": "WILD MAGENTA", "R": 210, "G": 50, "B": 130},
        {"Thread_No": 17747, "Thread_Name": "FESTIVAL PINK", "R": 220, "G": 60, "B": 140},
        {"Thread_No": 16153, "Thread_Name": "PETAL PINK", "R": 230, "G": 70, "B": 150},
    ],

    # 그룹 9: 26SS_22, 26
    "26SS_22": [
        {"Thread_No": 16073, "Thread_Name": "TOMATO RED", "R": 200, "G": 50, "B": 40},
        {"Thread_No": 16076, "Thread_Name": "JUICY RED", "R": 220, "G": 60, "B": 50},
        {"Thread_No": 13073, "Thread_Name": "INDIAN RED", "R": 180, "G": 40, "B": 30},
        {"Thread_No": 16080, "Thread_Name": "CHINA RED", "R": 240, "G": 70, "B": 60},
    ],
    "26SS_26": [...],
}

# ----------------------------
# 2. 추천 + 시각화 함수
# ----------------------------
def show_recommendations(ss_no, fabric_rgb=(15,30,60)):
    if ss_no not in recommendations:
        print(f"{ss_no}는 추천 리스트에 없음")
        return
    
    top_threads = recommendations[ss_no]
    
    fig, axs = plt.subplots(1, len(top_threads)+1, figsize=(12, 3))

    # 첫 번째: 원단색 배경
    rect = patches.Rectangle((0,0), 1,1, facecolor=np.array(fabric_rgb)/255)
    axs[0].add_patch(rect)
    axs[0].set_title("Fabric")
    axs[0].axis("off")

    # 추천 4개 컬러
    for ax, thread in zip(axs[1:], top_threads):
        rect = patches.Rectangle((0,0), 1,1, facecolor=np.array(fabric_rgb)/255)
        ax.add_patch(rect)

        # 글씨 = 해당 실컬러 RGB
        h_rgb = (thread["R"]/255, thread["G"]/255, thread["B"]/255)
        ax.text(0.5, 0.5, 'h',
                color=h_rgb,
                fontsize=100, ha='center', va='center', fontweight='bold')

        ax.set_title(f"{thread['Thread_No']}\n{thread['Thread_Name']}", fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    plt.show()

# ----------------------------
# 3. 실행 예시
# ----------------------------
if __name__ == "__main__":
    # 예: 26SS_08 → ELECTRIC GREY / SECRET BLUE / MAGIC BLUE / LOYAL BLUE
    show_recommendations("26SS_08", fabric_rgb=(191,80,74))
