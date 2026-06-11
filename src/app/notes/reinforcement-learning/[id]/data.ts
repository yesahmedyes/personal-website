/* eslint-disable @typescript-eslint/no-unsafe-return */
import { type ComponentType, type LazyExoticComponent } from "react";

interface Note {
  id: string;
  title: string;
  component: LazyExoticComponent<ComponentType>;
}

// Notes coming soon.
export const data: Note[] = [];
