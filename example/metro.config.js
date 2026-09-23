const path = require('path');
const { getDefaultConfig, mergeConfig } = require('@react-native/metro-config');

// react-native-shoe-tryon is `file:..`, so Metro has to watch the repository root to see edits to
// the library's source, and must resolve React from this app only — two copies of React in the
// bundle is a blank screen with no error.
const root = path.resolve(__dirname, '..');

module.exports = mergeConfig(getDefaultConfig(__dirname), {
  watchFolders: [root],
  resolver: {
    nodeModulesPaths: [path.join(__dirname, 'node_modules')],
  },
});
