"use client";

import { Add, HambergerMenu } from "iconsax-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";

export default function Navbar() {
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  const links = [
    {
      name: "Home",
      href: "/",
    },
    {
      name: "Blogs",
      href: "/blogs",
    },
    {
      name: "Notes",
      href: "/notes",
    },
  ];

  const pathname = usePathname();

  const isActiveLight = (path: string): string => {
    return pathname === path ? "text-orange-500" : "hover:text-textWhite";
  };

  const isActiveDark = (path: string): string => {
    return pathname === path ? "text-orange-500" : "hover:text-orange-500";
  };

  const toggleMenu = () => {
    setIsMenuOpen(!isMenuOpen);
  };

  return (
    <nav className={`flex flex-row text-textDarkGray font-normal py-6 lg:py-8 px-6`}>
      <button className="lg:hidden ml-auto" onClick={toggleMenu}>
        <HambergerMenu className={`stroke-textBlack`} />
      </button>

      <div className={`hidden lg:flex lg:flex-row gap-6 mx-auto`}>
        {links.map((link) => (
          <Link key={link.name} href={link.href} className={`cursor-pointer text-sm font-medium whitespace-nowrap ${isActiveDark(link.href)}`}>
            {link.name}
          </Link>
        ))}
      </div>

      <div className={`lg:hidden ${isMenuOpen ? "block" : "hidden"} fixed z-50 inset-0 bg-black/95 py-5 px-6 text-textLightGray`}>
        <div className="flex flex-col gap-6">
          <button className="ml-auto" onClick={() => setIsMenuOpen(false)}>
            <Add size={28} className={`rotate-45 stroke-textLightGray`} />
          </button>

          {links.map((link) => (
            <Link key={link.name} href={link.href} className={`cursor-pointer text-sm ${isActiveLight(link.href)}`} onClick={() => setIsMenuOpen(false)}>
              {link.name}
            </Link>
          ))}
        </div>
      </div>
    </nav>
  );
}
