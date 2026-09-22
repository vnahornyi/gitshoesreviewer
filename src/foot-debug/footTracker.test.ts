import {
  HOLD_MS,
  KEEP_SCORE,
  START_SCORE,
  trackedFeet,
  updateTracks,
  type FootTrack,
} from './footTracker';
import { framePoints } from './testPoints';

jest.mock('react-native-nitro-modules', () => ({ NitroModules: {} }));

const LEFT_FOOT = { heel: [0.4, 0.85], toe: [0.35, 0.95] } as {
  heel: [number, number];
  toe: [number, number];
};

function counter() {
  let id = 0;
  return () => ++id;
}

function run(frames: Array<{ time: number; points: number[] }>): FootTrack[] {
  const nextId = counter();
  return frames.reduce<FootTrack[]>(
    (tracks, frame) => updateTracks(tracks, frame.points, frame.time, nextId),
    [],
  );
}

describe('updateTracks', () => {
  it('starts a foot only above the start score', () => {
    expect(
      run([
        {
          time: 0,
          points: framePoints({
            left: { ...LEFT_FOOT, score: START_SCORE - 0.01 },
          }),
        },
      ]),
    ).toEqual([]);
    expect(
      run([
        {
          time: 0,
          points: framePoints({ left: { ...LEFT_FOOT, score: START_SCORE } }),
        },
      ]),
    ).toHaveLength(1);
  });

  it('keeps a started foot while its score stays above the keep score', () => {
    const tracks = run([
      { time: 0, points: framePoints({ left: { ...LEFT_FOOT, score: 0.5 } }) },
      {
        time: 1000,
        points: framePoints({ left: { ...LEFT_FOOT, score: KEEP_SCORE } }),
      },
    ]);
    expect(tracks).toHaveLength(1);
    expect(tracks[0].lastSeen).toBe(1000);
  });

  it('holds a lost foot, then drops it', () => {
    const seen = {
      time: 0,
      points: framePoints({ left: { ...LEFT_FOOT, score: 0.5 } }),
    };
    const empty = framePoints({});
    expect(run([seen, { time: HOLD_MS, points: empty }])).toHaveLength(1);
    expect(run([seen, { time: HOLD_MS + 1, points: empty }])).toEqual([]);
  });

  it('follows a foot by position when the model flips its side', () => {
    const tracks = run([
      { time: 0, points: framePoints({ left: { ...LEFT_FOOT, score: 0.5 } }) },
      {
        time: 33,
        points: framePoints({ right: { ...LEFT_FOOT, score: 0.5 } }),
      },
    ]);
    expect(tracks.map(track => [track.id, track.side])).toEqual([[1, 'right']]);
  });

  it('smooths small jitter', () => {
    const jitter = (dx: number) =>
      framePoints({
        left: { heel: [0.4 + dx, 0.85], toe: [0.35 + dx, 0.95], score: 0.5 },
      });
    const tracks = run([
      { time: 0, points: jitter(0) },
      { time: 33, points: jitter(0.01) },
    ]);
    const [foot] = trackedFeet(tracks, 33);
    expect(foot.heel?.x).toBeGreaterThan(0.4);
    expect(foot.heel?.x).toBeLessThan(0.405);
  });

  it('marks a held foot as stale', () => {
    const tracks = run([
      { time: 0, points: framePoints({ left: { ...LEFT_FOOT, score: 0.5 } }) },
      { time: 100, points: framePoints({}) },
    ]);
    expect(trackedFeet(tracks, 100)[0].stale).toBe(true);
  });
});
