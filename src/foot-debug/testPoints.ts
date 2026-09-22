type FootPoints = {
  heel: [number, number];
  toe: [number, number];
  score: number;
  ankle?: [number, number, number];
};

const HIDDEN: FootPoints = { heel: [0, 0], toe: [0, 0], score: 0 };
const NO_ANKLE = [0, 0, 0];

export function framePoints({
  left = HIDDEN,
  right = HIDDEN,
}: {
  left?: FootPoints;
  right?: FootPoints;
}): number[] {
  const foot = ({ heel, toe, score }: FootPoints) => ({
    toe: [...toe, score],
    smallToe: [toe[0], toe[1] + 0.01, score],
    heel: [...heel, score],
  });
  const l = foot(left);
  const r = foot(right);
  return [
    ...(left.ankle ?? NO_ANKLE),
    ...(right.ankle ?? NO_ANKLE),
    ...l.toe,
    ...l.smallToe,
    ...l.heel,
    ...r.toe,
    ...r.smallToe,
    ...r.heel,
  ];
}
