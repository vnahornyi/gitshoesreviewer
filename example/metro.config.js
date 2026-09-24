const path = require('path');
const { getDefaultConfig, mergeConfig } = require('@react-native/metro-config');

const root = path.resolve(__dirname, '..');
const library = require(path.join(root, 'package.json'));

// react-native-shoe-tryon is `file:..`, so its source is bundled from outside this app and Metro resolves its
// imports upward from the repository root — where the library keeps its own copies of react and react-native as
// devDependencies. Two copies of react in one bundle means the hook dispatcher is null and every component throws
// "Cannot read property 'useMemo' of null". So every shared package resolves to this app's copy, and the root's is
// blocked outright: a blockList makes the mistake impossible rather than merely unlikely.
const shared = Object.keys(library.peerDependencies ?? {});
const escape = (value) => value.replace(/[/\\^$*+?.()|[\]{}]/g, '\\$&');

const defaults = getDefaultConfig(__dirname);
const blocked = [defaults.resolver.blockList]
  .flat()
  .filter((pattern) => pattern instanceof RegExp);

module.exports = mergeConfig(defaults, {
  watchFolders: [root],
  resolver: {
    blockList: [
      ...blocked,
      ...shared.map(
        (name) =>
          new RegExp(
            `^${escape(path.join(root, 'node_modules', name))}\\${path.sep}.*$`,
          ),
      ),
    ],
    extraNodeModules: Object.fromEntries(
      shared.map((name) => [name, path.join(__dirname, 'node_modules', name)]),
    ),
  },
});
