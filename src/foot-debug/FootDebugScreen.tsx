import { useState } from 'react';
import {
  Pressable,
  StyleSheet,
  Text,
  View,
  type LayoutChangeEvent,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import type { FootModel } from 'react-native-foot-pose';
import { Camera, useCameraPermission } from 'react-native-vision-camera';
import { FootOverlay } from './FootOverlay';
import { ShoeLayer } from './ShoeLayer';
import type { SceneMode, Size } from './footAxes';
import { useFootPose } from './useFootPose';

type CameraPosition = 'back' | 'front';

const MODEL_LABELS: Record<FootModel, string> = {
  'rtmpose-m': 'RTMPose-m',
  'rtmw-x-l': 'RTMW x-l',
};

type Layer = 'axes' | 'shoe' | 'both';

const LAYER_LABELS: Record<Layer, string> = {
  axes: 'Стрілки',
  shoe: '3D',
  both: '3D + точки',
};

type Leg = 'matte' | 'cylinder';

const LEG_LABELS: Record<Leg, string> = {
  matte: 'Нога: маска',
  cylinder: 'Нога: циліндр',
};

type Look = 'camera' | 'clean';

const LOOK_LABELS: Record<Look, string> = {
  camera: 'Вигляд: камера',
  clean: 'Вигляд: чистий',
};

const SCENE_LABELS: Record<SceneMode, string> = {
  direct: 'На ноги',
  mirror: 'Дзеркало',
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

function Stats({
  status,
  fps,
  preprocessMs,
  inferenceMs,
  matteMs,
}: {
  status: string;
  fps: number;
  preprocessMs?: number;
  inferenceMs?: number;
  matteMs?: number;
}) {
  if (status !== 'ready') {
    return <Text style={styles.stats}>Модель: {status}</Text>;
  }
  return (
    <Text style={styles.stats}>
      {fps.toFixed(1)} fps · модель {inferenceMs?.toFixed(0) ?? '–'} мс ·
      підготовка {preprocessMs?.toFixed(0) ?? '–'} мс
      {matteMs === undefined ? '' : ` · маска ${matteMs.toFixed(0)} мс`}
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
  const [scene, setScene] = useState<SceneMode>('direct');
  const [position, setPosition] = useState<CameraPosition>('back');
  const [model, setModel] = useState<FootModel>('rtmpose-m');
  const [layer, setLayer] = useState<Layer>('axes');
  const [leg, setLeg] = useState<Leg>('matte');
  const [look, setLook] = useState<Look>('camera');
  const [view, setView] = useState<Size | null>(null);
  const showShoe = layer !== 'axes' && position === 'back';
  const { frameOutput, sample, feet, status, fps } = useFootPose(model, {
    legMatte: showShoe && leg === 'matte',
    sceneLight: showShoe && look === 'camera',
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
          device={position}
          isActive
          outputs={[frameOutput]}
          resizeMode="contain"
        />
        {sample && view && showShoe ? (
          <ShoeLayer
            feet={feet}
            sample={sample}
            view={view}
            legMatte={leg === 'matte'}
            matchCamera={look === 'camera'}
          />
        ) : null}
        {sample && view && layer !== 'shoe' ? (
          <FootOverlay
            feet={feet}
            frame={{ width: sample.frameWidth, height: sample.frameHeight }}
            view={view}
            scene={scene}
            flipX={position === 'front' && !sample.isMirrored}
          />
        ) : null}
      </View>
      <View style={[styles.panel, { paddingBottom: insets.bottom + 12 }]}>
        <Stats
          status={status}
          fps={fps}
          preprocessMs={sample?.preprocessMs}
          inferenceMs={sample?.inferenceMs}
          matteMs={sample?.matteMs}
        />
        <Toggle value={layer} options={LAYER_LABELS} onChange={setLayer} />
        {showShoe ? (
          <>
            <Toggle value={leg} options={LEG_LABELS} onChange={setLeg} />
            <Toggle value={look} options={LOOK_LABELS} onChange={setLook} />
          </>
        ) : null}
        <Toggle value={model} options={MODEL_LABELS} onChange={setModel} />
        <Toggle value={scene} options={SCENE_LABELS} onChange={setScene} />
        <Toggle
          value={position}
          options={{ back: 'Задня камера', front: 'Фронтальна' }}
          onChange={setPosition}
        />
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
