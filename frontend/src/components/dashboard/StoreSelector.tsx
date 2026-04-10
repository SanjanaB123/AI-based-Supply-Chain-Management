interface Props {
  stores: string[];
  selected: string;
  onChange: (store: string) => void;
  variant?: 'light' | 'dark';
}

function PinIcon({ dark }: { dark: boolean }) {
  return (
    <svg
      className={`h-3.5 w-3.5 shrink-0 ${dark ? 'text-slate-400' : 'text-slate-400'}`}
      viewBox="0 0 16 16"
      fill="currentColor"
      aria-hidden="true"
    >
      <path d="M8 1C5.24 1 3 3.24 3 6c0 3.75 5 9 5 9s5-5.25 5-9c0-2.76-2.24-5-5-5zm0 6.75A1.75 1.75 0 1 1 8 4.25a1.75 1.75 0 0 1 0 3.5z" />
    </svg>
  );
}

export default function StoreSelector({ stores, selected, onChange, variant = 'light' }: Props) {
  const isDark = variant === 'dark';

  return (
    <div
      className={`flex items-center gap-2 rounded-lg px-3 py-2 ${
        isDark
          ? 'border border-white/10 bg-white/5 hover:bg-white/8 transition-colors'
          : 'border border-slate-200 bg-white shadow-sm'
      }`}
    >
      <PinIcon dark={isDark} />
      <label
        htmlFor="store-select"
        className={`text-xs font-medium select-none ${isDark ? 'text-slate-400' : 'text-slate-400'}`}
      >
        Store
      </label>
      <select
        id="store-select"
        value={selected}
        onChange={e => onChange(e.target.value)}
        className={`border-0 bg-transparent text-sm font-semibold focus:outline-none focus:ring-0 cursor-pointer ${
          isDark ? 'text-slate-100' : 'text-slate-800'
        }`}
      >
        {stores.map(store => (
          <option key={store} value={store}>
            {store}
          </option>
        ))}
      </select>
    </div>
  );
}
