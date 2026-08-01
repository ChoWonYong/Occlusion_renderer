# KITTI 클래스 라벨링: 원본 규약 vs 이 레포의 처리

이 프로젝트는 KITTI Tracking의 라벨을 **그대로 쓰지 않고 4-class로 재정의**합니다. 어디가
어떻게 다른지, 각 차이가 왜 정당한지를 기록합니다. 2026-08-01 기준이며, 근거는 KITTI
devkit(`datasets/KITTI/devkit_tracking.zip` 안의 `devkit/readme.txt`)과 실제 라벨 집계입니다.

> **한 줄 요약:** 우리 수치는 **공개된 KITTI 벤치마크 수치와 직접 비교할 수 없습니다.**
> 클래스 집합·매핑·split·평가 코드가 모두 다릅니다. 우리 수치의 용도는 **동일 프로토콜 안에서
> arm 간 비교**이며, 그 목적에는 충분합니다.

---

## 1. 원본 KITTI가 정의하는 것

devkit `readme.txt:53-55`가 정의하는 타입은 9개입니다:

```
'Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc' or 'DontCare'
```

그런데 **KITTI 벤치마크가 실제로 채점하는 클래스는 2개뿐**입니다:

> Despite the fact that we have labeled 8 different classes, only the classes **'Car' and
> 'Pedestrian' are evaluated** in our benchmark, as only for those classes enough instances for a
> comprehensive evaluation have been labeled.

그리고 나머지를 어떻게 다루는지도 명시합니다:

> In our evaluation we only evaluate detections/objects **larger than 25 pixel (height)** in the
> image and **do not count Vans as false positives for cars or Sitting Persons as wrong positives
> for Pedestrians due to their similarity in appearance.** (All ignored objects are considered as
> DontCare areas.)

> Here, **'DontCare' labels denote regions in which objects have not been labeled**

즉 KITTI의 설계는 "**2개 클래스만 채점하고, 헷갈릴 만한 이웃 클래스와 미라벨 영역은 채점에서
제외**"입니다.

### 라벨 실측 (이 레포의 split 기준)

`configs/phase1_kitti.yaml`의 split은 KITTI train 21개 시퀀스를 train 12 / eval 9로 나눈
자체 split입니다(KITTI 공식 train/test 아님).

| 원본 타입 | train 박스 | eval 박스 |
|---|---:|---:|
| `Car` | 12,013 | 15,287 |
| `Van` | 1,174 | 2,127 |
| `Truck` | 402 | 787 |
| `Tram` | 178 | 417 |
| `Pedestrian` | 3,377 | 8,093 |
| `Person` (= `Person_sitting`) | 0 | 676 |
| `Cyclist` | 1,251 | 687 |
| `Misc` | 241 | 552 |
| `DontCare` | 8,772 | 9,267 |

> **철자 함정:** KITTI **Tracking**은 앉은 사람을 `Person`으로 적습니다. `Person_sitting`은
> object detection 벤치마크 쪽 철자이고 tracking 라벨에는 **한 번도 등장하지 않습니다**.
> `classes.kitti_map`에는 `Person_sitting`만 있어서 이 키는 실제로 아무것도 매칭하지 않습니다.

---

## 2. 우리의 4-class 정의

`configs/base.yaml`의 `classes`:

```yaml
names: [car, truck, person, bicycle]
kitti_map:
  Car: car
  Van: car
  Truck: truck
  Tram: truck
  Pedestrian: person
  Person_sitting: person   # tracking 라벨엔 없는 철자 — 실질 무효
  Cyclist: bicycle
```

매핑 결과:

| 우리 클래스 | train 박스 / 트랙 | eval 박스 / 트랙 |
|---|---:|---:|
| car | 13,187 / 307 | 17,414 / 329 |
| truck | 580 / 16 | 1,204 / 9 |
| person | 3,377 / 47 | 8,093 / 120 |
| bicycle | 1,251 / 18 | 687 / 19 |

**`kitti_map`은 파이프라인 전체의 단일 진실 공급원**입니다. 소비처:

| 파일 | 역할 |
|---|---|
| `data/kitti_tracking.py:120` | 매핑에 없는 타입을 **폐기** (`DontCare`, `Misc`, `Person`) |
| `data/kitti_tracking.py:81-83` | `dict.fromkeys(kitti_map.values())` 순서로 **category id 부여** |
| `data/build_phase1.py:249-251` | A/B 학습셋·eval셋 변환 |
| `synth/event_pipeline.py:320` | 합성 시 victim 클래스 판정 |
| `pool/build_kitti_sam3_pool.py:58,77,99` | occluder pool 클래스 결정·필터 |
| `data/build_kitti_finetune.py:18` | KITTI fine-tune split |

> **주의할 결합:** category id가 `kitti_map`에 **적힌 순서**로 정해지고, ByteTrack
> `MOTDataset`이 `cls = class_ids.index(category_id)`로 head 인덱스를 만듭니다. 평가 쪽은
> `classes.names` 순서로 인덱스를 잡습니다. 지금은 둘 다 `[car, truck, person, bicycle]`로
> 일치하지만, **`kitti_map`의 항목 순서만 바꿔도 조용히 클래스가 뒤섞입니다.**

---

## 3. 차이 항목별 정리와 근거

| 항목 | KITTI 공식 | 우리 | 근거 |
|---|---|---|---|
| 채점 클래스 수 | Car, Pedestrian **2종** | car, truck, person, bicycle **4종** | 자율주행 tracking에서 truck·bicycle도 관심 대상. KITTI가 2종만 채점하는 건 "충분한 인스턴스가 없어서"이지 무의미해서가 아님. 다만 그 희소성 때문에 우리 truck·bicycle 지표는 seed 분산이 큼(§5 참조) |
| `Van` | **ignore** (car의 FP로도 안 셈) | **car의 positive** | 학습 관점에서 Van은 car와 같은 검출 대상이고, eval 2,127박스를 버리면 car GT의 12%가 사라짐. **알고 한 이탈** — `eval/run_boxmot.py`의 `IGNORE_REGIONS`에 Van이 없는 이유를 테스트로 가드해 둠 |
| `Tram` | 채점 안 함 | **truck의 positive** | 4-class head에 tram을 위한 자리가 없고, 대형 궤도차량은 truck과 형태가 가장 가까움. 178/417박스로 소수 |
| `Cyclist` | 선택적 별도 클래스 | **bicycle** | 이름만 bicycle이고 실체는 KITTI 정의(사람+자전거 통째). **COCO의 bicycle(자전거만)과 의미가 다름** — COCO-pretrained 참조 모델과 비교할 때 유의 |
| `Person` (앉은 사람) | **ignore** (Pedestrian의 FP로 안 셈) | **평가에서 ignore** (2026-08-01 도입) | KITTI 사유("외형 유사성")가 우리 person 클래스에도 그대로 성립. §4 참조 |
| `DontCare` | **ignore** (미라벨 영역) | **평가에서 ignore** (2026-08-01 도입) | 라벨이 없는 영역에서 맞힌 것을 틀렸다고 세는 건 명백한 왜곡. §4 참조 |
| `Misc` | 채점 안 함 | **폐기 → FP 유발** | 2026-08-01 사용자 결정으로 현행 유지. §6 참조 |
| 25px 미만 객체 | 평가 제외 | **제외 안 함** | 작은 객체까지 전부 채점. 우리 쪽이 더 엄격하며, 모든 arm에 동일 적용되므로 비교에는 영향 없음 |
| split | 공식 train/test (test는 서버 제출) | train 21개를 12/9로 자체 분할 | 라벨이 공개된 구간에서만 평가해야 반복 실험이 가능. MOT 프로토콜 관례를 따름 |
| 평가 코드 | KITTI 전용 eval (CLEARMOT 계열) | TrackEval MOTChallenge 경로 (HOTA/CLEAR/Identity) | HOTA를 쓰기 위함. KITTI devkit은 CLEARMOT/MT-PT-ML 기준 |

### occluder pool은 별개 결정

합성 시 **붙이는** 객체는 car 0.65 / person 0.35의 **2-class**입니다. KITTI train에 `>=30`프레임
clean 연속 run이 truck 0개, bicycle 1개뿐이라 pool에서 제외했습니다. **detector head는 4-class를
그대로 유지**합니다(eval GT에 존재하므로). 자세한 근거는 README §1.

---

## 4. ignore 처리 (2026-08-01 도입)

### 왜 필요했나

`DontCare`와 `Person`은 `kitti_map`에 없어 변환 단계에서 폐기됐습니다
(`data/kitti_tracking.py:120`). 그러면 GT에 존재하지 않으므로 **FN이 되는 게 아니라**, 그 자리를
detector가 정확히 잡으면 매칭할 GT가 없어 **FP로 계산**됩니다. 잘 맞힐수록 손해입니다.

실측(`pastejit_mid` seed0, eval 9시퀀스): GT와 매칭되지 않은 검출 중

- **person: 53.8%가 DontCare 영역 안**, 5.6%가 앉은 사람 위
- **car: 51.6%가 DontCare 영역 안**

즉 미매칭 검출의 **절반 이상**이 "채점하지 말라고 표시된 자리"였습니다.

### 구현

`eval/run_boxmot.py`의 `IGNORE_REGIONS`:

```python
IGNORE_REGIONS = {
    "DontCare": None,          # None = 전 클래스
    "Person": ("person",),
    "Person_sitting": ("person",),
}
```

- 원본 KITTI 라벨에서 **직접** 읽습니다(`_ignore_regions_by_frame`). 변환기를 거치면 학습셋
  빌더 앞에도 놓여 train 데이터가 오염되기 때문입니다.
- GT 파일에 **distractor 클래스(id 8)** 로 기록하고 `--DO_PREPROC True`로 실행합니다.
- TrackEval 전처리는 검출을 **모든 GT 행(distractor 포함)과 Hungarian 매칭한 뒤 distractor에
  배정된 것만 제거**합니다. 실제 객체에 매칭된 검출은 절대 지워지지 않으므로, "영역에 걸치면
  무조건 버리기"가 만들 수 있는 오제거가 원천 차단됩니다.

### 왜 `Person`은 person 클래스에만

평가가 클래스별로 분리된 벤치마크(`KDS_CAR`, `KDS_TRUCK`, `KDS_PERSON`, `KDS_BICYCLE`)라
ignore 영역도 **어느 클래스의 GT에 쓸지**를 정해야 합니다. 실제 기록량:

| GT 파일 | ignore 행 | 내역 |
|---|---:|---|
| KDS_CAR / KDS_TRUCK / KDS_BICYCLE | 9,267 | DontCare만 |
| KDS_PERSON | 9,943 | DontCare 9,267 + Person 676 |

근거는 KITTI가 밝힌 사유 자체입니다 — "**due to their similarity in appearance**". 앉은 사람과
서 있는 보행자는 사람 검출기 입장에서 라벨 경계 문제지 실력 문제가 아닙니다. 반면 **car
검출기가 앉은 사람을 차라고 부른 건 진짜 오류**라 FP가 맞습니다. KITTI는 Car/Pedestrian만
채점하므로 "앉은 사람이 truck에 대해 무엇인가"를 정할 일이 없었고, 이 범위 결정은 우리 몫입니다.

데이터도 두 클래스가 실제로 다름을 보여줍니다:

| 타입 | 개수 | 3D 높이 중앙값 | 2D 박스 높이 중앙값 |
|---|---:|---:|---:|
| `Pedestrian` | 11,470 | **1.76 m** | 103 px |
| `Person` | 676 | **1.26 m** | 106 px |
| `Cyclist` | 1,938 | 1.73 m | 70 px |

키는 50cm 작은데 화면상 크기는 비슷합니다 — 앉아 있어 실제 키는 작지만 카메라에 더 가깝게
잡힌 경우가 많고, 2D 검출기가 구분하기 어려운 이유이기도 합니다.

### 효과 (기존 예측 재사용, `pastejit_mid` seed0)

| 클래스 | HOTA | MOTA | CLR_FP |
|---|---|---|---|
| car | 62.98 → **63.64** | 67.45 → **68.84** | 1,741 → 1,487 |
| person | 46.16 → **46.38** | 53.71 → **54.92** | 1,001 → 886 |
| truck·bicycle | 변화 없음 | 변화 없음 | 변화 없음 |

TP가 car 12·person 17개 줄었습니다(FN 동수 증가). 실제 객체와 DontCare 영역에 동시에 걸친
검출이 전역 Hungarian 최적해에서 DontCare 쪽에 배정되면 제거되는, TrackEval 전처리의 알려진
성질입니다(전체의 0.09%). 표준 동작이라 그대로 둡니다.

**모든 arm에 동일하게 적용되므로 arm 간 비교 결론은 바뀌지 않습니다.** 바뀌는 건 절대 수치이며,
`run_summary.json`의 `ignore_regions`·`trackeval_do_preproc` 필드로 어느 프로토콜에서 나온
수치인지 구분할 수 있습니다.

---

## 5. COCO 참조 모델과의 매핑

`--model-source coco-pretrained`로 COCO-pretrained YOLOX-X를 참조로 돌릴 때는
`eval/run_boxmot.py`의 `COCO_TO_PROJECT_CLASS`가 COCO 80-class **인덱스**를 우리 head 인덱스로
접습니다:

| COCO idx | COCO 이름 | → 우리 | 비고 |
|---:|---|---|---|
| 0 | person | person | |
| 1 | bicycle | bicycle | **의미 불일치** — COCO는 자전거만, 우리(=KITTI Cyclist)는 사람+자전거 |
| 2 | car | car | |
| 3 | motorcycle | bicycle | 이륜차를 한 클래스로 |
| 5 | bus | truck | |
| 7 | truck | truck | |

나머지 74개 COCO 클래스는 버립니다.

> 과거 `configs/base.yaml`에 `classes.coco_map`(이름→이름)이 있었으나 레거시
> `synth/video_pipeline.py` 한 곳에서만 쓰이는 **죽은 설정**이었고, COCO 참조 평가와는 무관했습니다.
> 2026-08-01에 제거했습니다. COCO 매핑을 바꿔야 한다면 `COCO_TO_PROJECT_CLASS` 한 곳만 고치면 됩니다.

---

## 6. 미결 항목

### `Misc` — 현행 유지 (2026-08-01 사용자 결정)

`Misc`는 8개 클래스 어디에도 안 맞는 나머지 차량류입니다. 실제 인스턴스를 확인한 결과
**관광버스·캠핑카(모터홈)·카라반 트레일러·박스형 화물 트레일러**였습니다:

| 시퀀스 | 트랙 | 3D h×w×l | 실체 |
|---|---|---|---|
| 0020 | 93 | 3.48 × 2.57 × 11.68 m | 대형 관광버스 |
| 0020 | 32 | 2.64 × 2.22 × 5.52 m | 캠핑카 |
| 0007 | 18 | 2.87 × 2.13 × 5.80 m | 자전거 실은 캠핑카 |
| 0020 | 30 | 2.41 × 2.03 × 5.13 m | 카라반 트레일러 |
| 0008 | 3 | 1.97 × 2.66 × 6.21 m | 트럭 적재함 |

`DontCare`와 달리 **위치와 3D 치수까지 정식으로 라벨링된 객체**입니다. eval 552박스 중 441개가
시퀀스 0020에 몰려 있습니다. 현재는 폐기되어 FP를 유발하며, COCO 매핑이 `bus → truck`이라
detector가 실제로 잡을 가능성이 있는 대상입니다.

세 선택지 — ① 현행 유지(FP), ② ignore, ③ truck의 positive로 매핑(train 241박스 포함 → **재학습
필요**) — 중 **①을 유지**하기로 했습니다. 재고할 때는 truck 지표의 불안정성과 함께 볼 것.

### 25px 필터

KITTI는 높이 25px 미만 객체를 평가에서 제외합니다. 우리는 적용하지 않습니다. 도입하면 truck·
bicycle처럼 작고 먼 객체가 많은 클래스의 수치가 올라갈 여지가 있습니다.

---

## 7. 이 문서를 읽고 하지 말아야 할 것

- 우리 HOTA/MOTA를 **KITTI 리더보드 수치와 나란히 놓기.** §3의 차이가 하나라도 남아 있는 한
  성립하지 않습니다.
- `classes.kitti_map`의 **항목 순서 바꾸기.** category id와 head 인덱스가 거기서 나옵니다(§2).
- ignore 처리 **전후 수치 섞어 쓰기.** `run_summary.json`의 `trackeval_do_preproc`로 구분하세요.
