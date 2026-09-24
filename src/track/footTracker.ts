import {
  footAxes,
  type FootAxis,
  type FootPoints,
  type FootSide,
  type Point,
} from './footAxes';
import {
  startOneEuro,
  stepOneEuro,
  type OneEuroParams,
  type OneEuroState,
} from './oneEuro';

export const START_SCORE = 0.3;
export const KEEP_SCORE = 0.15;
export const HOLD_MS = 300;
export const MAX_JUMP = 0.25;

// Points are normalized to the frame, so a speed of 1 is one frame width per second.
const SMOOTHING: OneEuroParams = {
  minCutoff: 1.5,
  beta: 6,
  derivativeCutoff: 1,
};

type SmoothedPoint = { x: OneEuroState; y: OneEuroState };

const OPTIONAL_POINTS = ['smallToe', 'ankle', 'heel'] as const;
type OptionalPoint = (typeof OPTIONAL_POINTS)[number];

// How long FootNet's points are carried by the body model before the body model's own points take over.
export const CARRY_MS = 600;

// Where FootNet put each point relative to the body model's big toe, which both models have on every frame.
type Offsets = { toe: Point } & Partial<Record<OptionalPoint, Point>>;

export type FootTrack = {
  id: number;
  side: FootSide;
  toe: SmoothedPoint;
  lastSeen: number;
  refinedAt: number;
  offsets?: Offsets;
} & Partial<Record<OptionalPoint, SmoothedPoint>>;

export type TrackedFoot = FootPoints & {
  id: number;
  side: FootSide;
  stale: boolean;
};

function startPoint(point: Point, time: number): SmoothedPoint {
  return { x: startOneEuro(point.x, time), y: startOneEuro(point.y, time) };
}

function stepPoint(
  state: SmoothedPoint,
  point: Point,
  time: number,
): SmoothedPoint {
  return {
    x: stepOneEuro(state.x, point.x, time, SMOOTHING),
    y: stepOneEuro(state.y, point.y, time, SMOOTHING),
  };
}

function valueOf(point: SmoothedPoint): Point {
  return { x: point.x.value, y: point.y.value };
}

function pointsOf(track: FootTrack): FootPoints {
  const points: FootPoints = { toe: valueOf(track.toe) };
  for (const name of OPTIONAL_POINTS) {
    const state = track[name];
    if (state) {
      points[name] = valueOf(state);
    }
  }
  return points;
}

// Toe and back of the foot: the heel if seen, else the ankle, else the toe alone.
function centre(foot: FootPoints): Point {
  const back = foot.heel ?? foot.ankle ?? foot.toe;
  return { x: (back.x + foot.toe.x) / 2, y: (back.y + foot.toe.y) / 2 };
}

function distance(track: FootTrack, foot: FootAxis): number {
  const a = centre(pointsOf(track));
  const b = centre(foot);
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function shift(point: Point, by: Point, sign: number): Point {
  return { x: point.x + sign * by.x, y: point.y + sign * by.y };
}

function offsetsOf(foot: FootAxis): Offsets {
  const offsets: Offsets = { toe: shift(foot.toe, foot.anchor, -1) };
  for (const name of OPTIONAL_POINTS) {
    const point = foot[name];
    if (point) {
      offsets[name] = shift(point, foot.anchor, -1);
    }
  }
  return offsets;
}

function atAnchor(offsets: Offsets, anchor: Point): FootPoints {
  const points: FootPoints = { toe: shift(offsets.toe, anchor, 1) };
  for (const name of OPTIONAL_POINTS) {
    const offset = offsets[name];
    if (offset) {
      points[name] = shift(offset, anchor, 1);
    }
  }
  return points;
}

// FootNet runs a few times a second, so on the frames in between its points are moved by how far the body model's big
// toe moved. Taking the body model's own points there instead would swap the heel for a worse one several times a
// second, which the shoe would show as a twitch.
function measure(
  track: FootTrack | null,
  foot: FootAxis,
  now: number,
): { points: FootPoints; offsets?: Offsets; refinedAt: number } {
  if (foot.refined) {
    return { points: foot, offsets: offsetsOf(foot), refinedAt: now };
  }
  const carried = track?.offsets;
  if (carried && now - track.refinedAt <= CARRY_MS) {
    return {
      points: atAnchor(carried, foot.anchor),
      offsets: carried,
      refinedAt: track.refinedAt,
    };
  }
  return { points: foot, refinedAt: track?.refinedAt ?? 0 };
}

// Smooth the points seen now; a point the model lost is dropped rather than held, so it cannot drift.
function stepFoot(
  track: FootTrack | null,
  foot: FootAxis,
  now: number,
): Omit<FootTrack, 'id' | 'lastSeen'> {
  const { points, offsets, refinedAt } = measure(track, foot, now);
  const next: Omit<FootTrack, 'id' | 'lastSeen'> = {
    side: foot.side,
    refinedAt,
    offsets,
    toe: track
      ? stepPoint(track.toe, points.toe, now)
      : startPoint(points.toe, now),
  };
  for (const name of OPTIONAL_POINTS) {
    const point = points[name];
    const state = track?.[name];
    if (point) {
      next[name] = state
        ? stepPoint(state, point, now)
        : startPoint(point, now);
    }
  }
  return next;
}

function matches(
  tracks: FootTrack[],
  feet: FootAxis[],
): Array<[FootTrack, FootAxis]> {
  const pairs = tracks
    .flatMap(track =>
      feet.map(foot => ({ track, foot, gap: distance(track, foot) })),
    )
    .filter(pair => pair.gap <= MAX_JUMP)
    .sort((a, b) => a.gap - b.gap);
  const matched: Array<[FootTrack, FootAxis]> = [];
  for (const { track, foot } of pairs) {
    if (matched.some(([t, f]) => t === track || f === foot)) {
      continue;
    }
    matched.push([track, foot]);
  }
  return matched;
}

// Tracks feet by position rather than by the model's left/right label, which flips between frames.
export function updateTracks(
  tracks: FootTrack[],
  points: readonly number[],
  refined: readonly number[] | undefined,
  now: number,
  nextId: () => number,
): FootTrack[] {
  const feet = footAxes(points, KEEP_SCORE, refined);
  const matched = matches(tracks, feet);

  const updated = tracks.flatMap(track => {
    const pair = matched.find(([t]) => t === track);
    if (!pair) {
      return now - track.lastSeen <= HOLD_MS ? [track] : [];
    }
    return [{ id: track.id, ...stepFoot(track, pair[1], now), lastSeen: now }];
  });

  const started = feet
    .filter(foot => foot.score >= START_SCORE)
    .filter(foot => !matched.some(([, f]) => f === foot))
    .slice(0, Math.max(0, 2 - updated.length))
    .map(foot => ({
      id: nextId(),
      ...stepFoot(null, foot, now),
      lastSeen: now,
    }));

  return [...updated, ...started];
}

export function trackedFeet(tracks: FootTrack[], now: number): TrackedFoot[] {
  return tracks.map(track => ({
    id: track.id,
    side: track.side,
    ...pointsOf(track),
    stale: track.lastSeen < now,
  }));
}
