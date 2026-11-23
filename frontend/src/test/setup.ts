import '@testing-library/jest-dom';
import { vi } from 'vitest';

// Mock ResizeObserver which is used by Recharts
global.ResizeObserver = vi.fn().mockImplementation(() => ({
  observe: vi.fn(),
  unobserve: vi.fn(),
  disconnect: vi.fn(),
}));

// Mock URL.createObjectURL / revokeObjectURL (jsdom doesn't implement these)
global.URL.createObjectURL = vi.fn(() => 'blob:mock-url');
global.URL.revokeObjectURL = vi.fn();

// Mock window.speechSynthesis for FarmerMode
if (typeof window !== 'undefined') {
  window.speechSynthesis = {
    speak: vi.fn(),
    cancel: vi.fn(),
    pause: vi.fn(),
    resume: vi.fn(),
    getVoices: vi.fn().mockReturnValue([]),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
    pending: false,
    speaking: false,
    paused: false,
    onvoiceschanged: null,
  } as unknown as SpeechSynthesis;

  // @ts-expect-error Mock SpeechSynthesisUtterance constructor
  window.SpeechSynthesisUtterance = vi.fn().mockImplementation((text: string) => ({
    text,
    voice: null,
    rate: 1,
    pitch: 1,
    volume: 1,
    onend: null,
    onerror: null,
  }));
}
