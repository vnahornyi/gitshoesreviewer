import { footAxes, type FootAxis, type FootSide, type Point } from './footAxes';
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

export type FootTrack = {
  id: number;
  side: FootSide;
  heel: SmoothedPoint;
  toe: SmoothedPoint;
  lastSeen: number;
};

export type TrackedFoot = {
  id: number;
  side: FootSide;
  heel: Point;
  toe: Point;
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

function middle(heel: Point, toe: Point): Point {
  return { x: (heel.x + toe.x) / 2, y: (heel.y + toe.y) / 2 };
}

function distance(track: FootTrack, foot: FootAxis): number {
  const a = middle(valueOf(track.heel), valueOf(track.toe));
  const b = middle(foot.heel, foot.toe);
  return Math.hypot(a.x - b.x, a.y - b.y);
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
  now: number,
  nextId: () => number,
): FootTrack[] {
  const feet = footAxes(points, KEEP_SCORE);
  const matched = matches(tracks, feet);

  const updated = tracks.flatMap(track => {
    const pair = matched.find(([t]) => t === track);
    if (!pair) {
      return now - track.lastSeen <= HOLD_MS ? [track] : [];
    }
    const foot = pair[1];
    return [
      {
        ...track,
        side: foot.side,
        heel: stepPoint(track.heel, foot.heel, now),
        toe: stepPoint(track.toe, foot.toe, now),
        lastSeen: now,
      },
    ];
  });

  const started = feet
    .filter(foot => foot.score >= START_SCORE)
    .filter(foot => !matched.some(([, f]) => f === foot))
    .slice(0, Math.max(0, 2 - updated.length))
    .map(foot => ({
      id: nextId(),
      side: foot.side,
      heel: startPoint(foot.heel, now),
      toe: startPoint(foot.toe, now),
      lastSeen: now,
    }));

  return [...updated, ...started];
}

export function trackedFeet(tracks: FootTrack[], now: number): TrackedFoot[] {
  return tracks.map(track => ({
    id: track.id,
    side: track.side,
    heel: valueOf(track.heel),
    toe: valueOf(track.toe),
    stale: track.lastSeen < now,
  }));
}
