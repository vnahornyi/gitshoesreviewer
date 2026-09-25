import { useEffect, useState } from 'react';
import {
  Pressable,
  StyleSheet,
  Text,
  View,
  type LayoutChangeEvent,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Camera, useCameraPermission } from 'react-native-vision-camera';
import {
  REFINED_MIN_SCORE,
  FOOT_JOINTS,
  ShoeLayer,
  frameToView,
  useFootPose,
  type Size,
} from 'react-native-shoe-tryon';
import { FootOverlay } from './FootOverlay';

type Layer = 'axes' | 'shoe' | 'both' | 'mask';

const CROP_COLORS = ['#3ddc84', '#ff9f0a'] as const;
const CROP_LABELS = ['Л', 'П'] as const;
const RTMPOSE_SEED_MIN_SCORE = 0.2;
const RTMPOSE_FOOT_JOINT_NAMES = [
  ['leftAnkle', 'leftBigToe', 'leftSmallToe', 'leftHeel'],
  ['rightAnkle', 'rightBigToe', 'rightSmallToe', 'rightHeel'],
] as const;
const RTMPOSE_FOOT_JOINTS = RTMPOSE_FOOT_JOINT_NAMES.map(names =>
  names.map(name => FOOT_JOINTS.indexOf(name)),
);

type LastRtmposeSearch = { points: number[]; at: number };

const LAYER_LABELS: Record<Layer, string> = {
  axes: 'Стрілки',
  shoe: '3D',
  both: '3D + точки',
  mask: 'Маска',
};

function Toggle<T extends string>({
  value,
  options,
  onChange,
}: {
  value: T;
  options: Record<T, string>;
  onChange: (next: T) => void;
}) {
  return (
    <View style={styles.toggle} accessibilityRole="radiogroup">
      {(Object.keys(options) as T[]).map(option => (
        <Pressable
          key={option}
          accessibilityRole="radio"
          accessibilityState={{ selected: option === value }}
          onPress={() => onChange(option)}
          style={[styles.option, option === value && styles.optionSelected]}
        >
          <Text style={styles.optionText}>{options[option]}</Text>
        </Pressable>
      ))}
    </View>
  );
}

// How many of FootNet's 16 points (8 per foot) it is sure of right now: the plain measure of whether our own model
// is carrying the pose or RTMPose is.
function sureOf(refined?: readonly number[]): number {
  if (!refined) {
    return 0;
  }
  let sure = 0;
  for (let point = 2; point < refined.length; point += 3) {
    if (refined[point] >= REFINED_MIN_SCORE) {
      sure += 1;
    }
  }
  return sure;
}

function Stats({
  status,
  fps,
  searchMs,
  matteMs,
  maskPreviewMs,
  maskPreviewStatus,
  refineMs,
  refined,
  crops,
  frame,
}: {
  refineMs?: number;
  refined?: number[];
  crops?: number[];
  frame?: Size;
  status: string;
  fps: number;
  searchMs?: number;
  matteMs?: number;
  maskPreviewMs?: number;
  maskPreviewStatus?: string;
}) {
  if (status !== 'ready') {
    return <Text style={styles.stats}>Модель: {status}</Text>;
  }
  return (
    <Text style={styles.stats}>
      {fps.toFixed(1)} fps · стопи {refineMs?.toFixed(0) ?? '–'} мс{' '}
      {sureOf(refined)}
      /16 · яскр. {crops?.[3]?.toFixed(2) ?? '–'}/{crops?.[7]?.toFixed(2) ?? '–'} ·
      пошук {searchMs?.toFixed(0) ?? '–'} мс
      {matteMs === undefined ? '' : ` · людина ${matteMs.toFixed(0)} мс`}
      {maskPreviewStatus === undefined
        ? ''
        : ` · маска ${maskPreviewStatus} ${
            maskPreviewMs?.toFixed(1) ?? '–'
          } мс`}
      {frame ? ` · ${frame.width}×${frame.height}` : ''}
    </Text>
  );
}

function CropBounds({
  crops,
  frame,
  view,
}: {
  crops: number[];
  frame: Size;
  view: Size;
}) {
  return (
    <View style={styles.cropOverlay} pointerEvents="none">
      {([0, 1] as const).map(index => {
        const offset = index * 4;
        const x = crops[offset];
        const y = crops[offset + 1];
        const side = crops[offset + 2];
        if (
          !Number.isFinite(x) ||
          !Number.isFinite(y) ||
          !Number.isFinite(side) ||
          side <= 0
        ) {
          return null;
        }
        const topLeft = frameToView({ x, y }, frame, view);
        const bottomRight = frameToView(
          { x: x + side, y: y + (side * frame.width) / frame.height },
          frame,
          view,
        );
        return (
          <View
            key={index}
            style={[
              styles.cropBounds,
              {
                left: topLeft.x,
                top: topLeft.y,
                width: bottomRight.x - topLeft.x,
                height: bottomRight.y - topLeft.y,
                borderColor: CROP_COLORS[index],
              },
            ]}
          >
            <Text style={[styles.cropLabel, { color: CROP_COLORS[index] }]}>
              {CROP_LABELS[index]}
            </Text>
          </View>
        );
      })}
    </View>
  );
}

function RtmposeSearchStatus({ search }: { search: LastRtmposeSearch | null }) {
  if (!search) {
    return <Text style={styles.hint}>RTMPose: очікується перший пошук</Text>;
  }
  const count = (indices: number[]) =>
    indices.filter(
      index => search.points[index * 3 + 2] >= RTMPOSE_SEED_MIN_SCORE,
    ).length;
  const [left, right] = RTMPOSE_FOOT_JOINTS;
  return (
    <Text style={styles.hint}>
      RTMPose, останній пошук ≥0.2: Л {count(left)}/4 · П {count(right)}/4 ·{' '}
      {((Date.now() - search.at) / 1000).toFixed(1)} с тому
    </Text>
  );
}

function PermissionGate({ onRequest }: { onRequest: () => void }) {
  return (
    <View style={styles.center}>
      <Text style={styles.message}>Потрібен доступ до камери</Text>
      <Pressable
        accessibilityRole="button"
        onPress={onRequest}
        style={styles.option}
      >
        <Text style={styles.optionText}>Дозволити</Text>
      </Pressable>
    </View>
  );
}

function LiveFeet() {
  const insets = useSafeAreaInsets();
  const [layer, setLayer] = useState<Layer>('axes');
  const [threshold, setThreshold] = useState<'0.3' | '0.5' | '0.7'>('0.5');
  const [view, setView] = useState<Size | null>(null);
  const [lastRtmposeSearch, setLastRtmposeSearch] =
    useState<LastRtmposeSearch | null>(null);
  const showMask = layer === 'mask';
  const showShoe = layer === 'shoe' || layer === 'both';
  const { frameOutput, sample, feet, status, fps } = useFootPose({
    legMatte: showShoe,
    sceneLight: showShoe,
    maskPreview: showMask,
    maskThreshold: Number(threshold),
  });

  useEffect(() => {
    if (sample && sample.inferenceMs > 0) {
      setLastRtmposeSearch({ points: sample.points, at: Date.now() });
    }
  }, [sample]);

  const onLayout = (event: LayoutChangeEvent) => {
    const { width, height } = event.nativeEvent.layout;
    setView({ width, height });
  };

  const renderCropBounds = () => {
    if (!showMask || !sample || !view) {
      return null;
    }
    return (
      <CropBounds
        crops={sample.crops}
        frame={{ width: sample.frameWidth, height: sample.frameHeight }}
        view={view}
      />
    );
  };

  const renderRtmposeSearchStatus = () => {
    if (!showMask) {
      return null;
    }
    return <RtmposeSearchStatus search={lastRtmposeSearch} />;
  };

  return (
    <View style={styles.screen}>
      <View style={styles.camera} onLayout={onLayout}>
        <Camera
          style={StyleSheet.absoluteFill}
          device="back"
          isActive
          outputs={[frameOutput]}
          resizeMode="contain"
        />
        {sample && view && (showShoe || showMask) ? (
          <ShoeLayer
            feet={showMask ? [] : feet}
            sample={sample}
            view={view}
            legMatte={showShoe}
            matchCamera={showShoe}
            maskPreview={showMask}
          />
        ) : null}
        {renderCropBounds()}
        {sample && view && (layer === 'axes' || layer === 'both') ? (
          <FootOverlay
            feet={feet}
            frame={{ width: sample.frameWidth, height: sample.frameHeight }}
            view={view}
          />
        ) : null}
      </View>
      <View style={[styles.panel, { paddingBottom: insets.bottom + 12 }]}>
        <Stats
          status={status}
          fps={fps}
          searchMs={sample?.searchMs}
          matteMs={showShoe ? sample?.matteMs : undefined}
          maskPreviewMs={showMask ? sample?.maskPreviewMs : undefined}
          maskPreviewStatus={showMask ? sample?.maskPreviewStatus : undefined}
          refineMs={sample?.refineMs}
          refined={sample?.refined}
          crops={sample?.crops}
          frame={
            sample
              ? { width: sample.frameWidth, height: sample.frameHeight }
              : undefined
          }
        />
        {renderRtmposeSearchStatus()}
        <Toggle value={layer} options={LAYER_LABELS} onChange={setLayer} />
        {showMask ? (
          <>
            <Toggle
              value={threshold}
              options={{ '0.3': '0.3', '0.5': '0.5', '0.7': '0.7' }}
              onChange={setThreshold}
            />
            <Text style={styles.hint}>
              Маска шукається лише в кропах, які знайшов RTMPose. Для запуску
              потрібна локальна модель.
            </Text>
          </>
        ) : null}
      </View>
    </View>
  );
}

export function FootDebugScreen() {
  const { hasPermission, requestPermission } = useCameraPermission();
  if (!hasPermission) {
    return <PermissionGate onRequest={requestPermission} />;
  }
  return <LiveFeet />;
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: 'black' },
  camera: { flex: 1 },
  cropOverlay: StyleSheet.absoluteFill,
  cropBounds: {
    position: 'absolute',
    borderWidth: 2,
  },
  cropLabel: {
    alignSelf: 'flex-start',
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    fontSize: 12,
    fontWeight: '700',
    paddingHorizontal: 4,
  },
  panel: { gap: 10, paddingHorizontal: 16, paddingTop: 12 },
  stats: { color: 'white', fontVariant: ['tabular-nums'], textAlign: 'center' },
  hint: { color: '#ccc', fontSize: 12, textAlign: 'center' },
  toggle: { flexDirection: 'row', gap: 8, justifyContent: 'center' },
  option: {
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderRadius: 20,
    backgroundColor: '#333',
  },
  optionSelected: { backgroundColor: '#0a84ff' },
  optionText: { color: 'white', fontWeight: '600' },
  center: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    gap: 16,
    backgroundColor: 'black',
  },
  message: { color: 'white', fontSize: 17 },
});
