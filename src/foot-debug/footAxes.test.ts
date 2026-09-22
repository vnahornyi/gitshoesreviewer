import { footAxes, frameToView, sideInScene } from './footAxes';
import { framePoints } from './testPoints';

jest.mock('react-native-nitro-modules', () => ({ NitroModules: {} }));

describe('footAxes', () => {
  it('returns the seen points with the model sides', () => {
    const points = framePoints({
      left: { heel: [0.4, 0.85], toe: [0.35, 0.95], score: 0.9 },
      right: { heel: [0.6, 0.85], toe: [0.65, 0.95], score: 0.8 },
    });
    expect(footAxes(points, 0.25)).toEqual([
      {
        side: 'left',
        toe: { x: 0.35, y: 0.95 },
        smallToe: { x: 0.35, y: 0.96 },
        heel: { x: 0.4, y: 0.85 },
        score: 0.9,
      },
      {
        side: 'right',
        toe: { x: 0.65, y: 0.95 },
        smallToe: { x: 0.65, y: 0.96 },
        heel: { x: 0.6, y: 0.85 },
        score: 0.8,
      },
    ]);
  });

  it('drops a foot below the score threshold', () => {
    const points = framePoints({
      left: { heel: [0.4, 0.85], toe: [0.35, 0.95], score: 0.9 },
      right: { heel: [0.6, 0.85], toe: [0.65, 0.95], score: 0.2 },
    });
    expect(footAxes(points, 0.25).map(foot => foot.side)).toEqual(['left']);
  });
});

describe('sideInScene', () => {
  it('keeps the side when looking at the feet and swaps it in a mirror', () => {
    expect(sideInScene('left', 'direct')).toBe('left');
    expect(sideInScene('left', 'mirror')).toBe('right');
  });
});

describe('frameToView', () => {
  const frame = { width: 720, height: 1280 };

  it('letterboxes a taller view around the frame', () => {
    expect(
      frameToView({ x: 0.5, y: 0 }, frame, { width: 360, height: 800 }, false),
    ).toEqual({ x: 180, y: 80 });
  });

  it('pillarboxes a wider view and mirrors x when asked', () => {
    expect(
      frameToView({ x: 0, y: 1 }, frame, { width: 600, height: 640 }, true),
    ).toEqual({ x: 480, y: 640 });
  });
});
