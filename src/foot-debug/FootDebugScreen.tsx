import { useState } from 'react';
import {
  Pressable,
  StyleSheet,
  Text,
  View,
  type LayoutChangeEvent,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Camera, useCameraPermission } from 'react-native-vision-camera';
import { FootOverlay } from './FootOverlay';
import { ShoeLayer } from './ShoeLayer';
import { REFINED_MIN_SCORE, type Size } from './footAxes';
import { useFootPose } from './useFootPose';

type Layer = 'axes' | 'shoe' | 'both';

const LAYER_LABELS: Record<Layer, string> = {
  axes: 'Стрілки',
  shoe: '3D',
  both: '3D + точки',
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
}) {
  if (status !== 'ready') {
    return <Text style={styles.stats}>Модель: {status}</Text>;
  }
  return (
    <Text style={styles.stats}>
      {fps.toFixed(1)} fps · стопи {refineMs?.toFixed(0) ?? '–'} мс{' '}
      {sureOf(refined)}
      /16 · кроп {crops?.[3].toFixed(2) ?? '–'}/{crops?.[7].toFixed(2) ?? '–'} ·
      пошук {searchMs?.toFixed(0) ?? '–'} мс
      {matteMs === undefined ? '' : ` · маска ${matteMs.toFixed(0)} мс`}
      {frame ? ` · ${frame.width}×${frame.height}` : ''}
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
  const [view, setView] = useState<Size | null>(null);
  const showShoe = layer !== 'axes';
  const { frameOutput, sample, feet, status, fps } = useFootPose({
    legMatte: showShoe,
    sceneLight: showShoe,
  });

  const onLayout = (event: LayoutChangeEvent) => {
    const { width, height } = event.nativeEvent.layout;
    setView({ width, height });
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
        {sample && view && showShoe ? (
          <ShoeLayer
            feet={feet}
            sample={sample}
            view={view}
            legMatte
            matchCamera
          />
        ) : null}
        {sample && view && layer !== 'shoe' ? (
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
          refineMs={sample?.refineMs}
          refined={sample?.refined}
          crops={sample?.crops}
          frame={
            sample
              ? { width: sample.frameWidth, height: sample.frameHeight }
              : undefined
          }
        />
        <Toggle value={layer} options={LAYER_LABELS} onChange={setLayer} />
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
  panel: { gap: 10, paddingHorizontal: 16, paddingTop: 12 },
  stats: { color: 'white', fontVariant: ['tabular-nums'], textAlign: 'center' },
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
