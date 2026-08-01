import fs from "node:fs/promises";
import { Workbook, SpreadsheetFile } from "@oai/artifact-tool";

const sourcePath = process.argv[2] || "C:/Users/pablo.esparcia/.codex/attachments/03050f55-450a-4cf3-a697-fdb20f6a7648/pasted-text.txt";
const outputDir = process.argv[3] || "C:/Users/pablo.esparcia/Documents/PhD/Código/outputs/regresiones_is";
const outputName = process.argv[4] || "tabla_regresiones_IS.xlsx";
const outputPath = `${outputDir}/${outputName}`;
const text = await fs.readFile(sourcePath, "utf8");
const lines = text.split(/\r?\n/);
const dependentMatch = text.match(/===.*?\s([A-Za-z0-9_-]+)_t\s*~/);
const dependentVariable = dependentMatch?.[1] || "Mkt-RF";

const results = [];
let current = null;
for (const raw of lines) {
  const line = raw.trim();
  const hm = line.match(/^horizon\s+(\d+)/i);
  if (hm) {
    current = { horizon: Number(hm[1]) };
    continue;
  }
  const model = line.match(/===.*?~\s*([A-Za-z0-9_]+)_\(t-0\)/);
  if (model && current) {
    current.predictor = model[1];
    continue;
  }
  if (!current?.predictor) continue;
  const nums = line.match(/[-+]?\d+\.\d+/g);
  if (line.includes("const") && nums?.length >= 3) {
    [current.constCoef, current.constSE, current.constT] = nums.slice(0, 3).map(Number);
    current.constSig = (line.match(/\*{1,3}/) || [""])[0];
  } else if (line.includes(`${current.predictor}_lag0`) && nums?.length >= 3) {
    [current.coef, current.se, current.tstat] = nums.slice(0, 3).map(Number);
    current.sig = (line.match(/\*{1,3}/) || [""])[0];
  } else {
    const fit = line.match(/R\^2:\s*([-+]?\d+\.\d+)\s+Obs:\s*(\d+)/);
    if (fit && current.coef !== undefined) {
      current.r2 = Number(fit[1]);
      current.obs = Number(fit[2]);
      results.push(current);
      current = null;
    }
  }
}

if (!results.length) throw new Error("No se pudieron extraer regresiones del archivo fuente.");

const horizons = [...new Set(results.map(r => r.horizon))].sort((a, b) => a - b);
const predictors = [...new Set(results.map(r => r.predictor))];
const columnLetter = n => {
  let s = "";
  while (n > 0) {
    n--;
    s = String.fromCharCode(65 + (n % 26)) + s;
    n = Math.floor(n / 26);
  }
  return s;
};
const lastCol = columnLetter(horizons.length + 1);
const pretty = s => s.replaceAll("_", " ");
const lookup = new Map(results.map(r => [`${r.predictor}|${r.horizon}`, r]));

const wb = Workbook.create();
const summary = wb.worksheets.add("Resultados");
const detail = wb.worksheets.add("Detalle");
const notes = wb.worksheets.add("Notas");
summary.showGridLines = false;
detail.showGridLines = false;
notes.showGridLines = false;

const navy = "#17324D";
const teal = "#0F766E";
const paleTeal = "#E8F4F2";
const paleBlue = "#EDF3F8";
const gold = "#D9A441";
const light = "#F7F9FB";
const gray = "#5B6573";
const border = "#D5DCE3";

summary.getRange(`A1:${lastCol}1`).merge();
summary.getRange("A1").values = [[`Regresiones predictivas IS · ${dependentVariable}`]];
summary.getRange(`A1:${lastCol}1`).format = {
  fill: navy, font: { color: "#FFFFFF", bold: true, size: 18 },
  verticalAlignment: "center"
};
summary.getRange(`A1:${lastCol}1`).format.rowHeight = 34;
summary.getRange(`A2:${lastCol}2`).merge();
summary.getRange("A2").values = [["Predictor en t frente al rendimiento acumulado futuro · estimación OLS"]];
summary.getRange(`A2:${lastCol}2`).format = {
  fill: "#DCE7F0", font: { color: navy, italic: true, size: 10 },
  verticalAlignment: "center"
};
summary.getRange(`A2:${lastCol}2`).format.rowHeight = 23;

const significant = results.filter(r => r.sig).length;
const strongest = [...results].sort((a, b) => Math.abs(b.tstat) - Math.abs(a.tstat))[0];
const cards = [
  ["Especificaciones", results.length, "Horizontes", horizons.length],
  ["Significativas (10%)", significant, "Mayor |t|", Math.abs(strongest.tstat)],
];
summary.getRange("A4:D5").values = cards;
summary.getRange("A4:D5").format.borders = { preset: "outside", style: "thin", color: border };
summary.getRange("A4:D5").format.fill = light;
summary.getRange("A4:A5").format.font = { bold: true, color: gray };
summary.getRange("C4:C5").format.font = { bold: true, color: gray };
summary.getRange("B4:B5").format.font = { bold: true, color: teal, size: 12 };
summary.getRange("D4:D5").format.font = { bold: true, color: teal, size: 12 };
summary.getRange("D5").format.numberFormat = "0.000";
summary.getRange(`E4:${lastCol}4`).merge();
summary.getRange("E4").values = [["Modelo con mayor |t|"]];
summary.getRange(`E4:${lastCol}4`).format = { fill: gold, font: { bold: true, color: "#FFFFFF" } };
summary.getRange(`E5:${lastCol}5`).merge();
summary.getRange("E5").values = [[`${pretty(strongest.predictor)} · h=${strongest.horizon} (t=${strongest.tstat.toFixed(3)})`]];
summary.getRange(`E5:${lastCol}5`).format = { fill: "#FFF8E8", font: { color: navy } };

const headerRow = 8;
const estimateMatrix = [
  ["Predictor", ...horizons.map(h => `Horizonte ${h}`)],
  ...predictors.map(p => [
    pretty(p),
    ...horizons.map(h => {
      const r = lookup.get(`${p}|${h}`);
      return r ? `${r.coef.toFixed(4)}${r.sig}\n(${r.se.toFixed(4)})` : "—";
    })
  ])
];
const estEnd = headerRow + estimateMatrix.length - 1;
summary.getRange(`A${headerRow}:${lastCol}${estEnd}`).values = estimateMatrix;
summary.getRange(`A${headerRow}:${lastCol}${headerRow}`).format = {
  fill: teal, font: { color: "#FFFFFF", bold: true },
  horizontalAlignment: "center", verticalAlignment: "center",
  borders: { bottom: { style: "medium", color: teal } }
};
summary.getRange(`A${headerRow + 1}:A${estEnd}`).format = {
  fill: paleBlue, font: { bold: true, color: navy }, verticalAlignment: "center"
};
summary.getRange(`B${headerRow + 1}:${lastCol}${estEnd}`).format = {
  horizontalAlignment: "center", verticalAlignment: "center", wrapText: true,
  borders: { insideHorizontal: { style: "thin", color: border } }
};
for (let r = headerRow + 1; r <= estEnd; r++) {
  if ((r - headerRow) % 2 === 0) summary.getRange(`B${r}:${lastCol}${r}`).format.fill = light;
  summary.getRange(`A${r}:${lastCol}${r}`).format.rowHeight = 32;
}

const r2Start = estEnd + 3;
summary.getRange(`A${r2Start}:${lastCol}${r2Start}`).merge();
summary.getRange(`A${r2Start}`).values = [["Bondad de ajuste (R²)"]];
summary.getRange(`A${r2Start}:${lastCol}${r2Start}`).format = {
  fill: navy, font: { color: "#FFFFFF", bold: true, size: 12 }
};
const r2Matrix = [
  ["Predictor", ...horizons.map(h => `Horizonte ${h}`)],
  ...predictors.map(p => [
    pretty(p),
    ...horizons.map(h => lookup.get(`${p}|${h}`)?.r2 ?? null)
  ])
];
const r2Header = r2Start + 1;
const r2End = r2Header + r2Matrix.length - 1;
summary.getRange(`A${r2Header}:${lastCol}${r2End}`).values = r2Matrix;
summary.getRange(`A${r2Header}:${lastCol}${r2Header}`).format = {
  fill: "#49677F", font: { color: "#FFFFFF", bold: true },
  horizontalAlignment: "center"
};
summary.getRange(`A${r2Header + 1}:A${r2End}`).format = { fill: paleBlue, font: { bold: true, color: navy } };
summary.getRange(`B${r2Header + 1}:${lastCol}${r2End}`).format = { numberFormat: "0.00%", horizontalAlignment: "center" };
summary.getRange(`B${r2Header + 1}:${lastCol}${r2End}`).conditionalFormats.add("colorScale", {
  colors: ["#FFFFFF", paleTeal, "#59A89C"],
  thresholds: ["min", "50%", "max"]
});

summary.getRange(`A${r2End + 2}:${lastCol}${r2End + 3}`).merge();
summary.getRange(`A${r2End + 2}`).values = [[
  "Notas: errores estándar entre paréntesis. * p<0,10; ** p<0,05; *** p<0,01. " +
  "La hoja «Detalle» conserva coeficientes, errores estándar, estadísticos t, R² y observaciones como valores numéricos."
]];
summary.getRange(`A${r2End + 2}:${lastCol}${r2End + 3}`).format = {
  fill: "#FFF8E8", font: { color: gray, italic: true, size: 9 },
  wrapText: true, verticalAlignment: "center",
  borders: { preset: "outside", style: "thin", color: "#E9D7A5" }
};

summary.getRange("A:A").format.columnWidth = 28;
summary.getRange(`B:${lastCol}`).format.columnWidth = 16;
summary.freezePanes.freezeRows(headerRow);

const detailHeaders = [
  "Predictor", "Horizonte", "Coeficiente", "Error estándar", "t-stat",
  "Significancia", "R²", "Observaciones", "Constante", "SE constante", "t constante", "Sig. constante"
];
const detailRows = results.map(r => [
  r.predictor, r.horizon, r.coef, r.se, r.tstat, r.sig || "",
  r.r2, r.obs, r.constCoef, r.constSE, r.constT, r.constSig || ""
]);
detail.getRange(`A1:L${detailRows.length + 1}`).values = [detailHeaders, ...detailRows];
detail.getRange("A1:L1").format = {
  fill: navy, font: { color: "#FFFFFF", bold: true },
  horizontalAlignment: "center", verticalAlignment: "center", wrapText: true
};
detail.getRange(`A2:A${detailRows.length + 1}`).format.font = { color: navy };
detail.getRange(`B2:B${detailRows.length + 1}`).format.numberFormat = "0";
detail.getRange(`C2:E${detailRows.length + 1}`).format.numberFormat = "0.0000";
detail.getRange(`G2:G${detailRows.length + 1}`).format.numberFormat = "0.0000";
detail.getRange(`H2:H${detailRows.length + 1}`).format.numberFormat = "#,##0";
detail.getRange(`I2:K${detailRows.length + 1}`).format.numberFormat = "0.0000";
detail.getRange(`A2:L${detailRows.length + 1}`).format.borders = {
  insideHorizontal: { style: "thin", color: "#E5E9ED" }
};
detail.getRange(`F2:F${detailRows.length + 1}`).conditionalFormats.add("containsText", {
  text: "*", format: { fill: paleTeal, font: { bold: true, color: teal } }
});
detail.tables.add(`A1:L${detailRows.length + 1}`, true, "DetalleRegresiones").style = "TableStyleMedium2";
detail.getRange("A:A").format.columnWidth = 27;
detail.getRange("B:B").format.columnWidth = 11;
detail.getRange("C:E").format.columnWidth = 14;
detail.getRange("F:F").format.columnWidth = 13;
detail.getRange("G:G").format.columnWidth = 10;
detail.getRange("H:H").format.columnWidth = 13;
detail.getRange("I:L").format.columnWidth = 14;
detail.freezePanes.freezeRows(1);

notes.getRange("A1:F1").merge();
notes.getRange("A1").values = [["Ficha metodológica"]];
notes.getRange("A1:F1").format = { fill: navy, font: { color: "#FFFFFF", bold: true, size: 16 } };
notes.getRange("A3:B9").values = [
  ["Elemento", "Descripción"],
  ["Variable dependiente", `${dependentVariable} acumulado al horizonte indicado`],
  ["Regresor", "Variable de opciones contemporánea (lag 0)"],
  ["Horizontes", horizons.join(", ")],
  ["Especificaciones", results.length],
  ["Convención", "Coeficiente con error estándar entre paréntesis"],
  ["Significancia", "* p<0,10; ** p<0,05; *** p<0,01"],
];
notes.getRange("A3:B3").format = { fill: teal, font: { color: "#FFFFFF", bold: true } };
notes.getRange("A4:A9").format = { fill: paleBlue, font: { bold: true, color: navy } };
notes.getRange("A3:B9").format.borders = { preset: "outside", style: "thin", color: border };
notes.getRange("A:A").format.columnWidth = 24;
notes.getRange("B:B").format.columnWidth = 65;
notes.getRange("A3:B9").format.wrapText = true;

const inspect = await wb.inspect({
  kind: "table",
  range: `Resultados!A${headerRow}:${lastCol}${Math.min(estEnd, headerRow + 8)}`,
  include: "values,formulas",
  tableMaxRows: 10,
  tableMaxCols: horizons.length + 1
});
console.log(inspect.ndjson);
const errors = await wb.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 50 },
  summary: "formula error scan"
});
console.log(errors.ndjson);

for (const [sheetName, range, fileName] of [
  ["Resultados", `A1:${lastCol}${r2End + 3}`, "preview_resultados.png"],
  ["Detalle", `A1:L${Math.min(detailRows.length + 1, 25)}`, "preview_detalle.png"],
  ["Notas", "A1:F10", "preview_notas.png"],
]) {
  const preview = await wb.render({ sheetName, range, scale: 1.35, format: "png" });
  await fs.writeFile(`${outputDir}/${fileName}`, new Uint8Array(await preview.arrayBuffer()));
}

await fs.mkdir(outputDir, { recursive: true });
const xlsx = await SpreadsheetFile.exportXlsx(wb);
await xlsx.save(outputPath);
console.log(JSON.stringify({ outputPath, regressions: results.length, predictors: predictors.length, horizons }));
