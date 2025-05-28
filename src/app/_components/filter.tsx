import { Add } from "iconsax-react";
import { useState } from "react";
import { Separator } from "./separator";

import { Popover, PopoverContent, PopoverTrigger } from "./popover";
import { Badge } from "./badge";
import { Checkbox } from "./checkbox";
import { ScrollArea } from "./scrollArea";

type FilterOption = {
  label: string;
  count?: number;
  icon?: React.ReactNode;
};

interface FilterProps {
  label: string;
  options: FilterOption[];
  onChange: (options: FilterOption[]) => void;
}

export default function Filter({ label, options, onChange }: FilterProps) {
  const [selected, setSelected] = useState<FilterOption[]>([]);

  const handleSelect = (option: FilterOption) => {
    const isSelected = selected.some((item) => item.label === option.label);
    const newSelected = isSelected ? selected.filter((item) => item.label !== option.label) : [...selected, option];
    setSelected(newSelected);
    onChange(newSelected);
  };

  const truncateText = (text: string, maxLength: number) => {
    return text.length > maxLength ? `${text.substring(0, maxLength)}...` : text;
  };

  return (
    <Popover>
      <PopoverTrigger asChild>
        <div className="h-12 w-full lg:w-[400px] flex flex-row gap-1 justify-between items-center bg-white rounded-lg border cursor-pointer">
          <div className="flex flex-row gap-1.5 items-center pl-4 pr-6 py-2">
            <Add className="stroke-textDarkGray" size={20} />
            <div className="text-textDarkGray font-normal text-sm whitespace-nowrap">{label}</div>
          </div>

          {selected.length > 0 && (
            <>
              {selected.length <= 2 ? (
                <div className="flex flex-row gap-2 items-center pl-2 pr-3 overflow-hidden">
                  {selected.map((option) => (
                    <Badge variant="outline" key={option.label} className="truncate max-w-36">
                      {truncateText(option.label, 10)}
                    </Badge>
                  ))}
                </div>
              ) : (
                <div className="pl-2 pr-3">
                  <Badge>{selected.length} selected</Badge>
                </div>
              )}
            </>
          )}
        </div>
      </PopoverTrigger>
      <PopoverContent align="start" className="p-0 mt-1 bg-white">
        <ScrollArea>
          <div className="flex flex-col gap-2 py-2.5 max-h-72">
            {options.map((option) => (
              <div key={option.label} className="flex flex-row items-center gap-2.5 px-3 py-1.5 hover:bg-accent cursor-pointer" onClick={() => handleSelect(option)}>
                <Checkbox checked={selected.some((item) => item.label === option.label)} />

                {option.icon && <span className="flex-shrink-0">{option.icon}</span>}

                <span className="text-sm text-textDarkGray font-normal line-clamp-1">{option.label.charAt(0).toUpperCase() + option.label.slice(1)}</span>

                {option.count !== undefined && <span className="text-xs rounded-full px-2 py-0.5 text-textTertiary bg-secondary border ml-auto flex-shrink-0">{option.count}</span>}
              </div>
            ))}
          </div>
        </ScrollArea>
      </PopoverContent>
    </Popover>
  );
}
