import { footAxes, frameToView } from './footAxes';
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
        ankle: undefined,
        heel: { x: 0.4, y: 0.85 },
        score: 0.9,
        refined: false,
        anchor: { x: 0.35, y: 0.95 },
      },
      {
        side: 'right',
        toe: { x: 0.65, y: 0.95 },
        smallToe: { x: 0.65, y: 0.96 },
        ankle: undefined,
        heel: { x: 0.6, y: 0.85 },
        score: 0.8,
        refined: false,
        anchor: { x: 0.65, y: 0.95 },
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

describe('frameToView', () => {
  const frame = { width: 720, height: 1280 };

  it('letterboxes a taller view around the frame', () => {
    expect(
      frameToView({ x: 0.5, y: 0 }, frame, { width: 360, height: 800 }),
    ).toEqual({ x: 180, y: 80 });
  });

  it('pillarboxes a wider view', () => {
    expect(
      frameToView({ x: 0, y: 1 }, frame, { width: 600, height: 640 }),
    ).toEqual({ x: 120, y: 640 });
  });
});
