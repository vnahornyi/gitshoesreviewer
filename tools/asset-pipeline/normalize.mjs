import { Document, NodeIO } from '@gltf-transform/core';
import { ALL_EXTENSIONS } from '@gltf-transform/extensions';
import {
  clearNodeTransform,
  dedup,
  flatten,
  prune,
  simplify,
  textureCompress,
  transformMesh,
  weld,
} from '@gltf-transform/functions';
import { MeshoptSimplifier } from 'meshoptimizer';
import sharp from 'sharp';
import { statSync, writeFileSync } from 'node:fs';
import { parseArgs } from 'node:util';

const HEEL_REGION = 0.25;

const rotationY = angle => {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  return [c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, 0, 0, 0, 1];
};

const rotationX = angle => {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  return [1, 0, 0, 0, 0, c, s, 0, 0, -s, c, 0, 0, 0, 0, 1];
};

const translation = (x, y, z) => [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1];

const uniformScale = k => [k, 0, 0, 0, 0, k, 0, 0, 0, 0, k, 0, 0, 0, 0, 1];

const degrees = value => (Number(value) * Math.PI) / 180;

const meshesOf = document => document.getRoot().listMeshes();

const positionsOf = document =>
  meshesOf(document).flatMap(mesh =>
    mesh.listPrimitives().flatMap(primitive => {
      const accessor = primitive.getAttribute('POSITION');
      const points = [];
      for (let i = 0; i < accessor.getCount(); i++) {
        points.push(accessor.getElement(i, [0, 0, 0]));
      }
      return points;
    }),
  );

const triangleCount = document =>
  meshesOf(document).reduce(
    (sum, mesh) =>
      sum +
      mesh.listPrimitives().reduce((acc, primitive) => {
        const indices = primitive.getIndices();
        const count = indices ? indices.getCount() : primitive.getAttribute('POSITION').getCount();
        return acc + count / 3;
      }, 0),
    0,
  );

const boundsOf = points => {
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (const point of points) {
    for (let axis = 0; axis < 3; axis++) {
      min[axis] = Math.min(min[axis], point[axis]);
      max[axis] = Math.max(max[axis], point[axis]);
    }
  }
  return { min, max, size: max.map((value, axis) => value - min[axis]) };
};

const principalAngleXZ = points => {
  const n = points.length;
  const meanX = points.reduce((sum, p) => sum + p[0], 0) / n;
  const meanZ = points.reduce((sum, p) => sum + p[2], 0) / n;
  let cxx = 0;
  let czz = 0;
  let cxz = 0;
  for (const [x, , z] of points) {
    cxx += (x - meanX) ** 2;
    czz += (z - meanZ) ** 2;
    cxz += (x - meanX) * (z - meanZ);
  }
  return 0.5 * Math.atan2(2 * cxz, cxx - czz);
};

const heelIsAtFront = points => {
  const { min, max, size } = boundsOf(points);
  const band = size[2] * HEEL_REGION;
  let backTop = -Infinity;
  let frontTop = -Infinity;
  for (const [, y, z] of points) {
    if (z <= min[2] + band) backTop = Math.max(backTop, y);
    if (z >= max[2] - band) frontTop = Math.max(frontTop, y);
  }
  return frontTop > backTop;
};

const applyToAllMeshes = (document, matrix) => {
  for (const mesh of meshesOf(document)) transformMesh(mesh, matrix);
};

export async function normalizeDocument(document, options = {}) {
  const { triangles = 30000, flip = false, preRotateX = 0, preRotateY = 0 } = options;
  const warnings = [];

  await document.transform(flatten());
  for (const node of document.getRoot().listNodes()) {
    if (node.getMesh()) clearNodeTransform(node);
  }

  if (preRotateX) applyToAllMeshes(document, rotationX(degrees(preRotateX)));
  if (preRotateY) applyToAllMeshes(document, rotationY(degrees(preRotateY)));

  applyToAllMeshes(document, rotationY(principalAngleXZ(positionsOf(document)) - Math.PI / 2));

  if (heelIsAtFront(positionsOf(document)) !== flip) applyToAllMeshes(document, rotationY(Math.PI));

  const { min, max, size } = boundsOf(positionsOf(document));
  if (size[1] > size[2]) {
    warnings.push('model is taller than it is long — probably not upright, try --pre-rotate-x 90 or -90');
  }
  applyToAllMeshes(document, translation(-(min[0] + max[0]) / 2, -min[1], -min[2]));
  applyToAllMeshes(document, uniformScale(1 / size[2]));

  const before = triangleCount(document);
  if (before > triangles) {
    await MeshoptSimplifier.ready;
    await document.transform(
      weld(),
      simplify({ simplifier: MeshoptSimplifier, ratio: triangles / before, error: 0.001 }),
    );
  }

  for (const material of document.getRoot().listMaterials()) {
    if (material.getAlphaMode() !== 'OPAQUE') {
      warnings.push(`material "${material.getName()}" is ${material.getAlphaMode()}, JPEG textures drop alpha`);
    }
  }

  return { trianglesBefore: before, trianglesAfter: triangleCount(document), warnings };
}

async function normalizeFile(input, output, values) {
  const io = new NodeIO().registerExtensions(ALL_EXTENSIONS);
  const document = await io.read(input);
  const report = await normalizeDocument(document, {
    triangles: Number(values.triangles),
    flip: values.flip,
    preRotateX: Number(values['pre-rotate-x']),
    preRotateY: Number(values['pre-rotate-y']),
  });
  const size = Number(values['texture-size']);
  await document.transform(
    textureCompress({ encoder: sharp, targetFormat: 'jpeg', resize: [size, size] }),
    dedup(),
    prune(),
  );
  await io.write(output, document);

  const bytes = statSync(output).size;
  const summary = { input, output, ...report, bytes, heelAtOrigin: true, lengthUnits: 1 };
  writeFileSync(output.replace(/\.glb$/, '.normalize.json'), `${JSON.stringify(summary, null, 2)}\n`);
  console.log(JSON.stringify(summary, null, 2));
  if (bytes > 5 * 1024 * 1024) console.warn(`warning: ${bytes} bytes is above the 5 MB budget`);
}

function syntheticShoe({ heelAtPositiveX }) {
  const document = new Document();
  const buffer = document.createBuffer();
  const boxes = [
    { min: [0, 0, -0.5], max: [2.7, 0.3, 0.5] },
    heelAtPositiveX ? { min: [1.9, 0, -0.45], max: [2.7, 1.0, 0.45] } : { min: [0, 0, -0.45], max: [0.8, 1.0, 0.45] },
  ];
  const positions = [];
  const indices = [];
  const faces = [
    [0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3],
  ];
  for (const { min, max } of boxes) {
    const base = positions.length / 3;
    for (let i = 0; i < 8; i++) {
      positions.push(i & 4 ? max[0] : min[0], i & 2 ? max[1] : min[1], i & 1 ? max[2] : min[2]);
    }
    for (const [a, b, c, d] of faces) indices.push(base + a, base + b, base + c, base + a, base + c, base + d);
  }
  const primitive = document
    .createPrimitive()
    .setAttribute('POSITION', document.createAccessor().setType('VEC3').setArray(new Float32Array(positions)).setBuffer(buffer))
    .setIndices(document.createAccessor().setType('SCALAR').setArray(new Uint32Array(indices)).setBuffer(buffer));
  const node = document
    .createNode('shoe')
    .setMesh(document.createMesh().addPrimitive(primitive))
    .setRotation([0, Math.sin(0.3), 0, Math.cos(0.3)])
    .setTranslation([5, 2, -3]);
  document.createScene().addChild(node);
  return document;
}

async function selfTest() {
  const assert = (condition, message) => {
    if (!condition) throw new Error(`self-test failed: ${message}`);
  };
  const near = (a, b) => Math.abs(a - b) < 1e-3;

  for (const heelAtPositiveX of [false, true]) {
    const document = syntheticShoe({ heelAtPositiveX });
    const report = await normalizeDocument(document);
    const points = positionsOf(document);
    const { min, max } = boundsOf(points);
    const tallest = points.reduce((best, p) => (p[1] > best[1] ? p : best));

    assert(near(min[2], 0) && near(max[2], 1), `length not normalized to [0,1] (heelAtPositiveX=${heelAtPositiveX})`);
    assert(near(min[1], 0), 'sole is not at y=0');
    assert(near(min[0] + max[0], 0), 'model is not centered on x');
    assert(tallest[2] < 0.35, `heel is not at the origin end (heelAtPositiveX=${heelAtPositiveX})`);
    assert(report.warnings.length === 0, `unexpected warnings: ${report.warnings.join('; ')}`);
  }
  console.log('self-test passed');
}

const { values, positionals } = parseArgs({
  allowPositionals: true,
  options: {
    'self-test': { type: 'boolean', default: false },
    triangles: { type: 'string', default: '30000' },
    'texture-size': { type: 'string', default: '1024' },
    flip: { type: 'boolean', default: false },
    'pre-rotate-x': { type: 'string', default: '0' },
    'pre-rotate-y': { type: 'string', default: '0' },
  },
});

if (values['self-test']) {
  await selfTest();
} else if (positionals.length === 2) {
  await normalizeFile(positionals[0], positionals[1], values);
} else {
  console.error(
    'usage: node normalize.mjs <in.glb> <out.glb> [--triangles 30000] [--texture-size 1024] [--flip] [--pre-rotate-x deg] [--pre-rotate-y deg]\n       node normalize.mjs --self-test',
  );
  process.exit(1);
}
