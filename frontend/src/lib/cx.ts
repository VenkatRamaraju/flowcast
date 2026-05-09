type ClassValue = string | number | false | null | undefined;

export default function cx(...values: ClassValue[]): string {
  return values.filter(Boolean).join(" ");
}
