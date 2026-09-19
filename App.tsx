import { StatusBar } from 'react-native';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { FootDebugScreen } from './src/foot-debug/FootDebugScreen';

function App() {
  return (
    <SafeAreaProvider>
      <StatusBar barStyle="light-content" />
      <FootDebugScreen />
    </SafeAreaProvider>
  );
}

export default App;
