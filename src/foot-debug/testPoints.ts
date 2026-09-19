type FootPoints = {
  heel: [number, number];
  toe: [number, number];
  score: number;
};

const HIDDEN: FootPoints = { heel: [0, 0], toe: [0, 0], score: 0 };

export function framePoints({
  left = HIDDEN,
  right = HIDDEN,
}: {
  left?: FootPoints;
  right?: FootPoints;
}): number[] {
  const ankle = [0.5, 0.5, 0.9];
  const foot = ({ heel, toe, score }: FootPoints) => ({
    toe: [...toe, score],
    smallToe: [toe[0], toe[1] + 0.01, score],
    heel: [...heel, score],
  });
  const l = foot(left);
  const r = foot(right);
  return [
    ...ankle,
    ...ankle,
    ...l.toe,
    ...l.smallToe,
    ...l.heel,
    ...r.toe,
    ...r.smallToe,
    ...r.heel,
  ];
}
