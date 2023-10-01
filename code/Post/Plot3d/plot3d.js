// Function to handle file selection and plotting
function handleFileSelect(event) {
    const files = event.target.files;

    if (files.length > 0) {
        // Create an array to store data and traces for each file
        const dataTraces = [];

        // Function to generate a random color
        function getRandomColor() {
            return `#${Math.floor(Math.random()*16777215).toString(16)}`;
        }
        const color_dict = {
            'Charges': 'red',
            'Inner Domain': 'lightgreen',
            'Interface': 'purple',
            'Outer Domain': 'lightblue',
            'Outer Border': 'orange',
            'Experimental': 'cyan',
            'test': 'red'
        };
        
        // Function to create a trace for a file with a specific color
        function createTrace(data, label) {
            const color = color_dict[label] || getRandomColor();
        // Function to create a trace for a file
            const trace = {
                x: data.map(row => row.X),
                y: data.map(row => row.Y),
                z: data.map(row => row.Z),
                mode: 'markers',
                type: 'scatter3d',
                marker: {
                    size: 4,
                    opacity: 0.7,
                    color: color
                },
                name: label // Use the file name as the trace label
            };
            return trace;
        }

        // Loop through each selected file
        for (let i = 0; i < files.length; i++) {
            const file = files[i];
            const reader = new FileReader();

            reader.onload = function (e) {
                // Parse the CSV data when the file is loaded
                Papa.parse(e.target.result, {
                    header: true,
                    dynamicTyping: true,
                    complete: function (results) {
                        const data = results.data;
                        const fileName = file.name.replace(/\.[^/.]+$/, ''); // Remove file extension
                        const trace = createTrace(data, fileName);
                        dataTraces.push(trace);

                        // If all files have been processed, create the plot
                        if (dataTraces.length === files.length) {
                            const layout = {
                                scene: {
                                    xaxis: { title: 'X-Axis' },
                                    yaxis: { title: 'Y-Axis' },
                                    zaxis: { title: 'Z-Axis' },
                                },
                                margin: {
                                    l: 0,
                                    r: 0,
                                    b: 0,
                                    t: 0
                                }
                            };
                            Plotly.newPlot('scatter-plots', dataTraces, layout);
                        }
                    }
                });
            };

            // Read the selected file as text
            reader.readAsText(file);
        }
    }
}

// Attach the event listener to the file input element
document.getElementById('file-input').addEventListener('change', handleFileSelect, false);
