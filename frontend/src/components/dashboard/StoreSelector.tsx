interface Props {
  stores: string[];
  selected: string;
  onChange: (store: string) => void;
}

export default function StoreSelector({ stores, selected, onChange }: Props) {
  return (
    <div className="flex items-center gap-2">
      <label htmlFor="store-select" className="text-sm font-medium text-gray-500">
        Store
      </label>
      <select
        id="store-select"
        value={selected}
        onChange={e => onChange(e.target.value)}
        className="rounded-lg border border-gray-200 bg-white px-3 py-2 text-sm font-medium text-gray-900 shadow-sm focus:border-transparent focus:outline-none focus:ring-2 focus:ring-indigo-500"
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
