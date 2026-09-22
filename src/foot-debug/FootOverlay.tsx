import { StyleSheet, Text, View } from 'react-native';
import {
  frameToView,
  sideInScene,
  type Point,
  type SceneMode,
  type Size,
} from './footAxes';
import type { TrackedFoot } from './footTracker';

const SIDE_COLORS = { left: '#3ddc84', right: '#ff9f0a' } as const;
const SIDE_LABELS = { left: 'Л', right: 'П' } as const;
const DOT = 12;
const LINE = 4;

const STALE_OPACITY = 0.4;

type FootOverlayProps = {
  feet: TrackedFoot[];
  frame: Size;
  view: Size;
  scene: SceneMode;
  flipX: boolean;
};

function Axis({
  heel,
  toe,
  color,
}: {
  heel: Point;
  toe: Point;
  color: string;
}) {
  const length = Math.hypot(toe.x - heel.x, toe.y - heel.y);
  const angle = Math.atan2(toe.y - heel.y, toe.x - heel.x);
  return (
    <View
      style={[
        styles.line,
        {
          width: length,
          left: (heel.x + toe.x) / 2 - length / 2,
          top: (heel.y + toe.y) / 2 - LINE / 2,
          backgroundColor: color,
          transform: [{ rotate: `${angle}rad` }],
        },
      ]}
    />
  );
}

function Dot({
  at,
  color,
  label,
}: {
  at: Point;
  color: string;
  label?: string;
}) {
  return (
    <View
      style={[
        styles.dot,
        { left: at.x - DOT / 2, top: at.y - DOT / 2, borderColor: color },
      ]}
    >
      {label ? <Text style={[styles.label, { color }]}>{label}</Text> : null}
    </View>
  );
}

function Foot({
  foot,
  scene,
  toView,
}: {
  foot: TrackedFoot;
  scene: SceneMode;
  toView: (point: Point) => Point;
}) {
  const toe = toView(foot.toe);
  const back = foot.heel ?? foot.ankle;
  const extras = (['smallToe', 'ankle', 'heel'] as const).flatMap(name => {
    const point = foot[name];
    return point ? [{ name, at: toView(point) }] : [];
  });
  const side = sideInScene(foot.side, scene);
  const color = SIDE_COLORS[side];
  return (
    <View
      style={[StyleSheet.absoluteFill, foot.stale && styles.stale]}
      pointerEvents="none"
    >
      {back && <Axis heel={toView(back)} toe={toe} color={color} />}
      {extras.map(({ name, at }) => (
        <Dot key={name} at={at} color={color} />
      ))}
      <Dot at={toe} color={color} label={SIDE_LABELS[side]} />
    </View>
  );
}

export function FootOverlay({
  feet,
  frame,
  view,
  scene,
  flipX,
}: FootOverlayProps) {
  const toView = (point: Point) => frameToView(point, frame, view, flipX);
  return (
    <View style={StyleSheet.absoluteFill} pointerEvents="none">
      {feet.map(foot => (
        <Foot key={foot.id} foot={foot} scene={scene} toView={toView} />
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  stale: { opacity: STALE_OPACITY },
  line: {
    position: 'absolute',
    height: LINE,
    borderRadius: LINE / 2,
  },
  dot: {
    position: 'absolute',
    width: DOT,
    height: DOT,
    borderRadius: DOT / 2,
    borderWidth: 3,
    backgroundColor: 'white',
  },
  label: {
    position: 'absolute',
    left: DOT,
    top: -DOT,
    fontSize: 16,
    fontWeight: '700',
  },
});
