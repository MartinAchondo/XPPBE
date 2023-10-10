function handleFileSelect(event) {
    const files = event.target.files;

    if (files.length > 0) {
        const dataTraces = [];

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
        
        function createTrace(data, label) {
            const color = color_dict[label] || getRandomColor();

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
                name: label 
            };
            return trace;
        }

        for (let i = 0; i < files.length; i++) {
            const file = files[i];
            const reader = new FileReader();

            reader.onload = function (e) {
                Papa.parse(e.target.result, {
                    header: true,
                    dynamicTyping: true,
                    complete: function (results) {
                        const data = results.data;
                        const fileName = file.name.replace(/\.[^/.]+$/, '');
                        const trace = createTrace(data, fileName);
                        dataTraces.push(trace);

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

            reader.readAsText(file);
        }
    }
}

document.getElementById('file-input').addEventListener('change', handleFileSelect, false);
