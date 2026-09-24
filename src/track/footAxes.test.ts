import { footAxes, frameToView } from './footAxes';
import { framePoints, refinedPoints } from './testPoints';

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

  // FootNet follows each foot in its own crop, so on the frames where the body model did not run its points are all
  // there is, and they have to stand on their own.
  it('keeps a FootNet foot the body model did not see', () => {
    const feet = footAxes(
      framePoints({}),
      0.25,
      refinedPoints([{ toe: [0.35, 0.95], heel: [0.3, 0.8] }, null]),
    );
    expect(feet).toHaveLength(1);
    expect(feet[0]).toMatchObject({
      side: 'left',
      refined: true,
      toe: { x: 0.35, y: 0.95 },
      heel: { x: 0.3, y: 0.8 },
      anchor: { x: 0.35, y: 0.95 },
    });
  });

  // Looking down at your own feet hides the heel, so FootNet is unsure of it about half the time. Its toes are still
  // worth having, and the body model's ankle stands in for the back of the foot.
  it('keeps FootNet toes without its heel and takes the ankle from the body model', () => {
    const points = framePoints({
      left: {
        heel: [0.4, 0.85],
        toe: [0.35, 0.95],
        score: 0.9,
        ankle: [0.42, 0.8, 0.9],
      },
    });
    const [foot] = footAxes(
      points,
      0.25,
      refinedPoints([{ toe: [0.3, 0.9] }, null]),
    );
    expect(foot).toMatchObject({
      refined: true,
      toe: { x: 0.3, y: 0.9 },
      ankle: { x: 0.42, y: 0.8 },
    });
    expect(foot.heel).toBeUndefined();
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

  it('round-trips an off-centre point from a landscape frame into a portrait view', () => {
    const landscape = { width: 1472, height: 828 };
    const view = { width: 393, height: 852 };
    const point = { x: 0.75, y: 0.25 };
    const mapped = frameToView(point, landscape, view);

    expect(mapped.x).toBeCloseTo(294.75, 8);
    expect(mapped.y).toBeCloseTo(370.734375, 8);

    const scale = Math.min(view.width / landscape.width, view.height / landscape.height);
    const offsetX = (view.width - landscape.width * scale) / 2;
    const offsetY = (view.height - landscape.height * scale) / 2;
    expect((mapped.x - offsetX) / (landscape.width * scale)).toBeCloseTo(point.x, 10);
    expect((mapped.y - offsetY) / (landscape.height * scale)).toBeCloseTo(point.y, 10);
  });
});
