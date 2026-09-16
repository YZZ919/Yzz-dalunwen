import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const work = "D:/大论文实验/IR-VIO的复现";
const inputPath = `${work}/results/weighting_comparison.csv`;
const outputPath = `${work}/results/irvio_reproduction_summary.xlsx`;
const previewPath = `${work}/results/irvio_reproduction_summary_preview.png`;

function parseCsvLine(line) {
  const fields = [];
  let value = "";
  let quoted = false;
  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];
    if (char === '"') {
      if (quoted && line[i + 1] === '"') {
        value += '"';
        i += 1;
      } else {
        quoted = !quoted;
      }
    } else if (char === "," && !quoted) {
      fields.push(value);
      value = "";
    } else {
      value += char;
    }
  }
  fields.push(value);
  return fields;
}

const text = await fs.readFile(inputPath, "utf8");
const lines = text.trim().split(/\r?\n/);
const headers = [
  "Sequence", "Baseline ATE RMSE (m)", "Weighting ATE RMSE (m)",
  "ATE reduction (%)", "Baseline success", "Weighting success",
  "Weight rows", "Valid weight rows", "Alpha minimum", "Alpha mean",
  "Alpha maximum", "Beta mean", "Hybrid mean",
];
const numericColumns = new Set([1, 2, 3, 6, 7, 8, 9, 10, 11, 12]);
const booleanColumns = new Set([4, 5]);
const rows = lines.slice(1).filter(Boolean).map((line) =>
  parseCsvLine(line).map((value, index) => {
    if (booleanColumns.has(index)) return value.toLowerCase() === "true";
    if (numericColumns.has(index)) return value === "" ? null : Number(value);
    return value;
  }),
);

const workbook = Workbook.create();
const sheet = workbook.worksheets.add("Results");
sheet.showGridLines = false;
sheet.tabColor = "#1F4E78";

sheet.getRange("A2:M2").merge();
sheet.getRange("A2").values = [["EuRoC clean-room two-layer weighting results"]];
sheet.getRange("A2:M2").format.font = { name: "Arial", size: 14, bold: true, color: "#1F1F1F" };
sheet.getRange("A3:M3").format.borders = { bottom: { style: "thin", color: "#9EADBA" } };

sheet.getRange("A4:C4").values = [["Attempted sequences", "Pipeline success rate", "Weighting precision win rate"]];
sheet.getRange("A5").formulas = [[`=COUNTA(A9:A${8 + rows.length})`]];
sheet.getRange("B5").formulas = [[`=SUMPRODUCT(E9:E${8 + rows.length},F9:F${8 + rows.length})/A5`]];
sheet.getRange("C5").formulas = [[`=COUNTIFS(D9:D${8 + rows.length},">0")/A5`]];
sheet.getRange("A4:C4").format = {
  fill: "#D9EAF7",
  font: { name: "Arial", size: 10, bold: true, color: "#1F1F1F" },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  wrapText: true,
};
sheet.getRange("A4:C4").format.rowHeight = 28;
sheet.getRange("A5:C5").format = {
  font: { name: "Arial", size: 12, bold: true, color: "#1F4E78" },
  horizontalAlignment: "center",
  verticalAlignment: "center",
};
sheet.getRange("B5:C5").format.numberFormat = "0.0%";

sheet.getRange("A7:M7").merge();
sheet.getRange("A7").values = [["Positive reduction means the weighting run has lower SE(3)-aligned ATE. This is a clean-room implementation of equations (2)-(9), not the authors' full IR-VIO system."]];
sheet.getRange("A7:M7").format.font = { name: "Arial", size: 10, italic: true, color: "#595959" };

const tableMatrix = [headers, ...rows];
sheet.getRange("A8").write(tableMatrix);
const lastRow = 8 + rows.length;
sheet.getRange(`A8:M${lastRow}`).format.font = { name: "Arial", size: 10, color: "#1F1F1F" };
sheet.getRange("A8:M8").format = {
  fill: "#1F4E78",
  font: { name: "Arial", size: 10, bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  wrapText: true,
  borders: { insideVertical: { style: "thin", color: "#FFFFFF" } },
};
sheet.getRange(`B9:C${lastRow}`).format.numberFormat = "0.0000";
sheet.getRange(`D9:D${lastRow}`).format.numberFormat = "0.00";
sheet.getRange(`G9:H${lastRow}`).format.numberFormat = "#,##0";
sheet.getRange(`I9:M${lastRow}`).format.numberFormat = "0.000";
sheet.getRange(`D9:D${lastRow}`).conditionalFormats.add("cellIs", {
  operator: "greaterThan", formula: 0,
  format: { fill: "#E2F0D9", font: { color: "#375623" } },
});
sheet.getRange(`D9:D${lastRow}`).conditionalFormats.add("cellIs", {
  operator: "lessThan", formula: 0,
  format: { fill: "#FCE4D6", font: { color: "#9C0006" } },
});
sheet.freezePanes.freezeRows(8);

sheet.getRange(`A8:M${lastRow}`).format.autofitColumns();
sheet.getRange(`A8:A${lastRow}`).format.columnWidth = 18;
sheet.getRange(`B8:C${lastRow}`).format.columnWidth = 17;
sheet.getRange(`D8:D${lastRow}`).format.columnWidth = 18;
sheet.getRange(`E8:F${lastRow}`).format.columnWidth = 14;
sheet.getRange(`G8:M${lastRow}`).format.columnWidth = 13;
sheet.getRange("A7:M7").format.rowHeight = 30;

workbook.recalculate();
const inspected = await workbook.inspect({
  kind: "table", range: `Results!A1:M${lastRow}`,
  include: "values,formulas", tableMaxRows: 25, tableMaxCols: 13,
});
console.log(inspected.ndjson);
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 100 },
  summary: "final formula error scan",
});
console.log(errors.ndjson);

const preview = await workbook.render({ sheetName: "Results", range: `A1:M${lastRow}`, scale: 1.3, format: "png" });
await fs.writeFile(previewPath, new Uint8Array(await preview.arrayBuffer()));
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
const savedBlob = await FileBlob.load(outputPath);
const savedWorkbook = await SpreadsheetFile.importXlsx(savedBlob);
const savedCheck = await savedWorkbook.inspect({
  kind: "table", range: `Results!A1:M${lastRow}`,
  include: "values,formulas", tableMaxRows: 25, tableMaxCols: 13,
});
console.log(savedCheck.ndjson);
const savedPreview = await savedWorkbook.render({
  sheetName: "Results", range: `A1:M${lastRow}`, scale: 1.3, format: "png",
});
await fs.writeFile(previewPath, new Uint8Array(await savedPreview.arrayBuffer()));
console.log(outputPath);
